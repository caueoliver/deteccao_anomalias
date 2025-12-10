import streamlit as st
import os
import gdown
from PIL import Image
import numpy as np
import time

# Importação das pipelines
try:
    from pipelines.gradcam_pipeline import infer_gradcam
    from pipelines.pix2pix_pipeline import infer_pix2pix
except ImportError as e:
    st.error(f"❌ Erro crítico: Não foi possível importar as pipelines. Verifique a pasta 'pipelines'. Detalhe: {e}")
    st.stop()

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Detecção de Doenças",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ESTILO CSS ---
st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background-color: #0E1117 !important;
        color: #FAFAFA !important;
    }
    
    .main {
        background-color: #0E1117 !important;
    }

    
    /* Título Principal - Branco */
    h1 {
        color: #FFFFFF !important; 
        font-family: 'Helvetica Neue', sans-serif;
    }

    h3 {
        color: #E0E0E0 !important;
        border-bottom: 2px solid #4CAF50; /* Mantive a linha verde sutil, se quiser tirar, apague */
        padding-bottom: 10px;
    }

    p, span, label, div, li {
        color: #FAFAFA !important;
    }

    .stButton>button {
        background-color: #2E7D32 !important;
        color: white !important;
        border-radius: 8px;
        height: 50px;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1B5E20 !important;
    }

    .metric-card {
        background-color: #262730 !important; /* Fundo cinza chumbo */
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        text-align: center;
        border: 1px solid #444; /* Borda discreta */
    }
    </style>
    """, unsafe_allow_html=True)

PATH_GRADCAM = 'weights/gradcam_model_state_dict.pth'
PATH_PIX2PIX = 'weights/pix2pix_netG_ts.pt'

# IDs do Google Drive
GDRIVE_ID_GRADCAM = '1hWxOArHhUsiS-Dv-9tZdraZhRK6i0iaV'
GDRIVE_ID_PIX2PIX = '1MWMTTIKhuhcNYXMsGborWoWROhwKHEeF'

# --- FUNÇÃO DE DOWNLOAD ---
@st.cache_resource
def download_model_weights():
    # cria a pasta weights se ela não existir
    if not os.path.exists('weights'):
        os.makedirs('weights')

    # Download do GradCAM se não existir
    if not os.path.exists(PATH_GRADCAM):
        with st.spinner('Baixando modelo Grad-CAM do Google Drive... (Isso acontece apenas uma vez)'):
            url = f'https://drive.google.com/uc?id={GDRIVE_ID_GRADCAM}'
            gdown.download(url, PATH_GRADCAM, quiet=False)

    # Download do Pix2Pix se não existir
    if not os.path.exists(PATH_PIX2PIX):
        with st.spinner('Baixando modelo Pix2Pix do Google Drive...'):
            url = f'https://drive.google.com/uc?id={GDRIVE_ID_PIX2PIX}'
            gdown.download(url, PATH_PIX2PIX, quiet=False)

# --- CHAMADA INICIAL DO DOWNLOAD ---
# Tenta baixar tudo antes de continuar
try:
    download_model_weights()
except Exception as e:
    st.error(f"Erro ao baixar modelos: {e}")

# --- FUNÇÃO DE VERIFICAÇÃO ---
def check_models():
    missing = []
    if not os.path.exists(PATH_GRADCAM): 
        missing.append(f"Grad-CAM ({PATH_GRADCAM})")
    if not os.path.exists(PATH_PIX2PIX): 
        missing.append(f"Pix2Pix ({PATH_PIX2PIX})")
    return missing

# --- INTERFACE DO STREAMLIT ---
st.markdown("""
    <style>
    .stApp { background-color: #f0f2f6; }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/628/628283.png", width=80)
    st.title("Painel de Controle")
    st.markdown("---")
    uploaded_file = st.file_uploader("📂 Carregar Folha (JPG/PNG)", type=["jpg", "png", "jpeg"])

st.title("🌿 Diagnóstico Inteligente")
st.markdown("Sistema de detecção de patologias em plantas utilizando **Visão Computacional** e **IA Generativa**.")

# Verifica se o download funcionou
missing_models = check_models()
if missing_models:
    st.error("⚠️ ARQUIVOS DE MODELO NÃO ENCONTRADOS!")
    st.warning("O download automático falhou. Verifique se os IDs do Google Drive estão corretos e se os arquivos estão como 'Público/Qualquer pessoa com o link'.")
    st.write("Arquivos faltando:")
    for m in missing_models:
        st.code(m)
    st.stop()

if uploaded_file is None:
    st.info("👈 Por favor, faça o upload de uma imagem na barra lateral para começar a análise.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 1. Upload")
        st.caption("Envie foto da folha suspeita.")
    with col2:
        st.markdown("#### 2. Processamento")
        st.caption("IA analisa textura e cor.")
    with col3:
        st.markdown("#### 3. Diagnóstico")
        st.caption("Mapas de calor indicam a doença.")

else:
    img_pil = Image.open(uploaded_file).convert("RGB")
    
    col_orig, col_btn = st.columns([1, 2])
    with col_orig:
        st.image(img_pil, caption="Imagem Original", use_container_width=True)
    with col_btn:
        st.write("") 
        st.write("")
        analyze = st.button("🔍 INICIAR ANÁLISE COMPLETA", use_container_width=True)

    if analyze:
        with st.spinner("Rodando modelos neurais..."):
            time.sleep(0.5) 
            
            res_grad = infer_gradcam(img_pil, PATH_GRADCAM)
            
            res_pix = infer_pix2pix(img_pil, PATH_PIX2PIX)
            
            st.success("Análise concluída com sucesso!")
            
            tab1, tab2, tab3 = st.tabs(["📊 Classificação (Grad-CAM)", "🧬 Anomalia (Pix2Pix)", "📝 Relatório Técnico"])
            
            # ABA 1: GRAD-CAM
            with tab1:
                st.subheader("Classificação Supervisionada")
                st.markdown(
                    "O modelo **ResNet18** classifica a folha e o **Grad-CAM** destaca as regiões determinantes."
                )
                
                m_class = res_grad['metrics'].get('Predição', 'N/A')
                m_conf = res_grad['metrics'].get('Confiança', '0%')
                
                k1, k2 = st.columns(2)
                k1.metric("Diagnóstico", m_class, delta="Alerta" if m_class=="Doente" else "Normal", delta_color="inverse")
                k2.metric("Confiança do Modelo", m_conf)
                
                g1, g2 = st.columns(2)
                with g1:
                    st.image(res_grad['img2'], caption="Mapa de Calor (Ativação)", use_container_width=True)
                with g2:
                    st.image(res_grad['img1'], caption="Sobreposição na Folha", use_container_width=True)

            # ABA 2: PIX2PIX
            with tab2:
                st.subheader("Reconstrução Generativa")
                st.markdown(
                    "O modelo **Pix2Pix** tenta reconstruir uma versão saudável da folha. "
                    "As diferenças de cor (CIEDE2000) indicam a doença."
                )
                
                metrics_pix = res_pix.get('metrics', {})
                score_anomalia = metrics_pix.get('mean_anomaly_score', 0.0)
                max_erro = metrics_pix.get('max_anomaly_score', 0.0)
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Erro Médio (CIEDE2000)", f"{score_anomalia:.2f}")
                c2.metric("Pico de Anomalia", f"{max_erro:.2f}")
                c3.metric("Status Visual", "Anômalo" if score_anomalia > 1.3 else "Consistente") 
                
                st.markdown("---")
                
                p1, p2, p3 = st.columns(3)
                with p1:
                    st.image(res_pix['input_gray'], caption="Entrada (Visão da IA)", use_container_width=True)
                with p2:
                    st.image(res_pix['reconstructed'], caption="Tentativa de Reconstrução (Saudável)", use_container_width=True)
                with p3:
                    st.image(res_pix['deltae_heatmap'], caption="Mapa de Doença (Diferença)", use_container_width=True)

            with tab3:
                st.markdown("### Metodologia")
                st.info("""
                **1. Grad-CAM (Gradient-weighted Class Activation Mapping):**
                Técnica que utiliza os gradientes da última camada convolucional para entender quais partes da imagem levaram o modelo a classificar a planta como doente.
                
                **2. Pix2Pix (cGAN) + CIEDE2000:**
                Utiliza uma rede adversária generativa treinada apenas com folhas saudáveis. Ao receber uma folha doente, a rede falha em reconstruir as lesões (pintando-as de verde). A métrica de cor CIEDE2000 calcula a diferença perceptiva, isolando a doença.
                """)
                
                st.download_button(
                    label="📥 Baixar Relatório Completo (JSON)",
                    data=str({**res_grad['metrics'], **res_pix['metrics']}),
                    file_name="relatorio_diagnostico.json",
                    mime="application/json"
                )