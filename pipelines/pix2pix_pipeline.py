import torch
import numpy as np
from PIL import Image
from skimage.color import rgb2lab, deltaE_ciede2000
import matplotlib.cm as cm

class Pix2PixTorchScript:
    def __init__(self, ts_path: str, image_size: int = 256):
        # Configura GPU ou CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = image_size
        
        try:
            # Carrega o modelo salvo (.pt)
            self.netG = torch.jit.load(ts_path, map_location=self.device)
            self.netG.eval()
            print(f"✅ Modelo Pix2Pix carregado em: {self.device}")
        except Exception as e:
            print(f"❌ Erro ao carregar modelo Pix2Pix: {e}")
            raise e

    def infer(self, pil_img: Image.Image) -> dict:
        # 1. PREPARAÇÃO DA IMAGEM
        # Redimensiona para 256x256 (padrão da rede)
        img_rgb = pil_img.convert("RGB").resize((self.image_size, self.image_size), Image.BICUBIC)
        
        # Converte para Escala de Cinza (1 Canal)
        img_gray_L = img_rgb.convert("L") 
        
        # Transforma em Numpy [0, 1]
        input_np = np.array(img_gray_L).astype(np.float32) / 255.0
        
        # Cria tensor [1, 1, 256, 256] (Batch, Channel, Height, Width)
        input_tensor = torch.from_numpy(input_np).unsqueeze(0).unsqueeze(0)
        input_tensor = input_tensor.to(self.device)

        # 2. INFERÊNCIA (A IA "Pinta" a folha)
        with torch.no_grad():
            fake_tensor = self.netG(input_tensor)
        
        # 3. PÓS-PROCESSAMENTO (Matemática Ajustada)
        # Tira da GPU e converte para Numpy (Altura, Largura, Canais)
        fake_np = fake_tensor[0].cpu().permute(1, 2, 0).numpy()
        
        # IMPORTANTE: Apenas removemos valores negativos ou > 1.
        # Não dividimos por 2, pois seu modelo aprendeu a gerar direto no intervalo [0, 1].
        fake_np = np.clip(fake_np, 0, 1)
        
        # Converte para formato de imagem (0 a 255)
        fake_u8 = (fake_np * 255).astype(np.uint8)
        real_u8 = np.array(img_rgb)

        # 4. CÁLCULO DE ANOMALIA (CIEDE2000)
        # Converte RGB para Lab (necessário para o cálculo de percepção humana)
        real_lab = rgb2lab(real_u8)
        fake_lab = rgb2lab(fake_u8)
        
        # Calcula a diferença pixel a pixel
        diff_map = deltaE_ciede2000(real_lab, fake_lab)
        
        # 5. GERAR MAPA DE CALOR VISUAL
        # Normalização visual: Consideramos erro > 20 como anomalia grave (vermelho)
        heatmap_norm = np.clip(diff_map / 20.0, 0, 1)
        
        # Aplica o mapa de cores 'jet' (Azul=Ok, Vermelho=Erro)
        heatmap_colored = cm.get_cmap('jet')(heatmap_norm)[..., :3] 
        heatmap_u8 = (heatmap_colored * 255).astype(np.uint8)

        # Retorna dicionário pronto para o App
        return {
            "input_gray": img_gray_L.convert("RGB"),   # Apenas para mostrar na tela
            "reconstructed": Image.fromarray(fake_u8), # A folha verde gerada
            "deltae_heatmap": Image.fromarray(heatmap_u8), # O mapa de doença
            "metrics": {
                "max_anomaly_score": float(diff_map.max()),
                "mean_anomaly_score": float(diff_map.mean())
            }
        }

# --- SINGLETON PARA O STREAMLIT ---
# Evita carregar o modelo múltiplas vezes
_pix2pix = None

def infer_pix2pix(pil_img: Image.Image, ts_path: str) -> dict:
    global _pix2pix
    if _pix2pix is None:
        _pix2pix = Pix2PixTorchScript(ts_path=ts_path)
    
    return _pix2pix.infer(pil_img)