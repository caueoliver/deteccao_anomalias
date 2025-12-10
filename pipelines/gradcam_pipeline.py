import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import models, transforms
import cv2

# variável global para cache do modelo
_predictor = None

class GradCamPredictor:
    def __init__(self, weights_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # carregar a ResNet18
        self.model = models.resnet18(weights=None)
        # Ajustar a última camada para 2 classes (igual ao treino)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 2)
        
        # carregar os pesos salvos
        # map_location garante que funcione em qualquer PC
        state = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        # preparar os Hooks para o Grad-CAM
        self.target_layer = self.model.layer4[-1] # Última camada convolucional
        self.gradients = None
        self.activations = None

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
            
        def forward_hook(module, input, output):
            self.activations = output

        self.target_layer.register_full_backward_hook(backward_hook)
        self.target_layer.register_forward_hook(forward_hook)

        # transformações
        self.tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def infer(self, pil_img: Image.Image) -> dict:
        # preprocessamento
        img_tensor = self.tf(pil_img.convert("RGB")).unsqueeze(0).to(self.device)
        
        # zerar gradientes anteriores
        self.model.zero_grad()
        
        # forward
        output = self.model(img_tensor)
        probs = F.softmax(output, dim=1)
        pred_idx = output.argmax(dim=1).item()
        
        # backward (Para gerar o mapa)
        score = output[0, pred_idx]
        score.backward()
        
        # gerar mapa
        grads = self.gradients.cpu().data.numpy()[0]
        fmap = self.activations.cpu().data.numpy()[0]
        
        weights = np.mean(grads, axis=(1, 2))
        cam = np.zeros(fmap.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * fmap[i]
            
        cam = np.maximum(cam, 0)
        cam = cam / (cam.max() + 1e-8)
        cam = cv2.resize(cam, (224, 224))
        
        # preparar visualização
        img_np = np.array(pil_img.convert("RGB").resize((224, 224))).astype(np.float32) / 255.0
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) / 255.0
        
        overlay = 0.6 * img_np + 0.4 * heatmap_colored
        overlay = np.clip(overlay, 0, 1)
        
        # converter para PIL para exibir no App
        overlay_pil = Image.fromarray((overlay * 255).astype(np.uint8))
        heatmap_pil = Image.fromarray((heatmap_colored * 255).astype(np.uint8))

        return {
            "img1": overlay_pil,   # imagem com sobreposição
            "img2": heatmap_pil,   # apenas o calor
            "metrics": {
                "Predição": "Doente" if pred_idx == 1 else "Saudável",
                "Confiança": f"{probs[0, pred_idx].item()*100:.2f}%"
            }
        }

def infer_gradcam(pil_img: Image.Image, weights_path: str) -> dict:
    global _predictor
    if _predictor is None:
        _predictor = GradCamPredictor(weights_path=weights_path)
    return _predictor.infer(pil_img)