# 🌱 Plant Disease Anomaly Detection

> **Detecção não-supervisionada de doenças em plantas utilizando técnicas de reconstrução de imagens e análise de anomalias de cor.**

Este repositório contém a implementação de um sistema de detecção de anomalias para diagnóstico de doenças em plantas. O projeto foi desenvolvido como parte da avaliação da disciplina **Introdução a Inteligência Artificial** na **Universidade de Brasília (UnB)**.

O método baseia-se na premissa de que um modelo generativo treinado apenas com imagens de folhas saudáveis terá dificuldade em reconstruir regiões doentes (anômalas), permitindo a detecção da doença através do cálculo do erro de reconstrução (resíduo).

## Conceito Teórico

Inspirado no trabalho de *Katafuchi & Tokunaga (2021)*, o sistema utiliza a **reconstrutibilidade de cores**:
1.  **Treinamento:** O modelo aprende a distribuição de cores e formas de folhas saudáveis.
2.  **Inferência:** Ao processar uma folha doente, o modelo tenta "consertá-la" para parecer saudável.
3.  **Detecção:** A diferença entre a imagem original e a reconstruída gera um mapa de calor, destacando a lesão.

## Funcionalidades

* **Pré-processamento:** Normalização e preparação de imagens.
* **Modelo Generativo:** Implementação de rede neural para reconstrução de imagens (ex: Autoencoder / GAN / Pix2Pix).
* **Visualização de Anomalias:** Geração de mapas de calor (heatmaps) pixel a pixel baseados no erro de reconstrução (CIEDE2000 ou MSE).
* **Métricas de Avaliação:** Cálculo de pontuações de anomalia para classificar folhas como saudáveis ou doentes.

## Tecnologias Utilizadas

* **Linguagem:** Python 3.x
* **Deep Learning:** [PyTorch] 
* **Visão Computacional:** OpenCV, Pillow
* **Análise de Dados:** NumPy, Pandas, Matplotlib/Seaborn

##  Como Executar o Projeto

Siga os passos abaixo para configurar o ambiente e rodar a detecção:

```bash
git clone [https://github.com/caueoliver/deteccao_anomalias]
cd NOME_DO_REPOSITORIO
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt