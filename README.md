# ORACLE: market regime detector

<img width="576" height="447" alt="image" src="https://github.com/user-attachments/assets/12388386-fd74-446a-af75-2c97d2151d91" />

![Market Oracle](https://img.shields.io/badge/ORACLE-🔮-purple)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)
![AI](https://img.shields.io/badge/AI%20ML-transformers-orange)
![Open Source Love](https://badges.frapsoft.com/os/v1/open-source.svg?v=103) 

**ORACLE** is a multi-asset market regime detection system that uncovers hidden market states, visualizes regime shifts over time, and can provide early warnings for potentiel crises or bull runs

---

## features
-**multi asset awarness:** stocks, bonds, commodities, FX, and macro indicators  

-**latent market embddings:** variational autoencoders (VAE) or transformers to encode market states 

-**regime detection:** unsupervised clustering reveals hidden market regimes (bull, bear, crisis, recovery)

-**visualization dashboard:** interactive timeline, embeddings projection, and asset/macro overlays

-**predictive module:** forecast probabilities of upcoming regimes

-**scenario simulation:** explore "what-if" market events and their impact on regimes

---

## installation

1.**clone the repo:**

git clone https://github.com/Youcef3939/ORACLE.git

cd ORACLE

2.**create a virtual environment:**

python -m venv venv

source venv/bin/activate  # Linux/Mac

venv\Scripts\activate     # Windows

pip install -r requirements.txt

3.**explore data & prototypes in notebooks/** 

4.**run the dashboard:** 

streamlit run dashboard/app.py


---

## architecture
[data sources] --> [data pipeline] --> [feature engineering] --> [model core] --> [regime detection] --> [visualization & dashboard]


---

## data flow
raw Data → preprocessing → feature engineering → model core → latent embeddings → clustering → regime labels → visualization / alerts 


---

## future enhancments
. integrate more assets & micro indicators

. add transformer-based predictive regime forecasting

. real time alerts for early crisis detection

. improve clustering with temporal continuity ( for exemple HMMs, hidden markov models)


---

## output images
![alt text](<Capture d'écran 2025-09-09 020144.png>)

![alt text](<Capture d'écran 2025-09-09 020126.png>)

![alt text](<Capture d'écran 2025-09-09 020103.png>)
