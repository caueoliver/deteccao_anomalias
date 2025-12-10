import torch
import numpy as np
from PIL import Image
from skimage.color import rgb2lab, deltaE_ciede2000
import matplotlib.cm as cm

class Pix2PixTorchScript:
    def __init__(self, ts_path: str, image_size: int = 256):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = image_size
        
        try:
            # carrega o modelo salvo (.pt)
            self.netG = torch.jit.load(ts_path, map_location=self.device)
            self.netG.eval()
            print(f"Modelo Pix2Pix carregado em: {self.device}")
        except Exception as e:
            print(f"Erro ao carregar modelo Pix2Pix: {e}")
            raise e

    def infer(self, pil_img: Image.Image) -> dict:
        # PREPARAÇÃO DA IMAGEM
        # redimensiona para 256x256
        img_rgb = pil_img.convert("RGB").resize((self.image_size, self.image_size), Image.BICUBIC)
        
        # converte para escala de cinza 
        img_gray_L = img_rgb.convert("L") 
        
        # transforma em Numpy [0, 1]
        input_np = np.array(img_gray_L).astype(np.float32) / 255.0
        
        # cria tensor [1, 1, 256, 256] (Batch, Channel, Height, Width)
        input_tensor = torch.from_numpy(input_np).unsqueeze(0).unsqueeze(0)
        input_tensor = input_tensor.to(self.device)

        # INFERÊNCIA
        with torch.no_grad():
            fake_tensor = self.netG(input_tensor)
        
        # PÓS-PROCESSAMENTO
        # tira da GPU e converte para Numpy (Altura, Largura, Canais)
        fake_np = fake_tensor[0].cpu().permute(1, 2, 0).numpy()

        fake_np = np.clip(fake_np, 0, 1)
        
        # converte para formato de imagem (0 a 255)
        fake_u8 = (fake_np * 255).astype(np.uint8)
        real_u8 = np.array(img_rgb)

        # CÁLCULO DE ANOMALIA (CIEDE2000)
        # converte RGB para Lab (necessário para o cálculo de percepção humana)
        real_lab = rgb2lab(real_u8)
        fake_lab = rgb2lab(fake_u8)
        
        # calcula a diferença pixel a pixel
        diff_map = deltaE_ciede2000(real_lab, fake_lab)
        
        # GERAR MAPA DE CALOR VISUAL
        # normalização visual: Consideramos erro > 20 como anomalia grave (vermelho)
        heatmap_norm = np.clip(diff_map / 20.0, 0, 1)
        
        #aplica o mapa de cores 'jet' 
        heatmap_colored = cm.get_cmap('jet')(heatmap_norm)[..., :3] 
        heatmap_u8 = (heatmap_colored * 255).astype(np.uint8)

        #retorna dicionário pronto para o App
        return {
            "input_gray": img_gray_L.convert("RGB"),  
            "reconstructed": Image.fromarray(fake_u8), # a folha verde gerada
            "deltae_heatmap": Image.fromarray(heatmap_u8), # o mapa de doença
            "metrics": {
                "max_anomaly_score": float(diff_map.max()),
                "mean_anomaly_score": float(diff_map.mean())
            }
        }


# evita carregar o modelo múltiplas vezes
_pix2pix = None

def infer_pix2pix(pil_img: Image.Image, ts_path: str) -> dict:
    global _pix2pix
    if _pix2pix is None:
        _pix2pix = Pix2PixTorchScript(ts_path=ts_path)
    
    return _pix2pix.infer(pil_img)