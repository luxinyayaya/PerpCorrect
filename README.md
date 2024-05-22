# Perplexity-aware Correction (PerpCorrect) for Robust Alignment with Noisy Preferences

Code for paper "Perplexity-aware Correction for Robust Alignment with Noisy Preferences"

## Robust alignment with EYE

~~~bash
pip install -r requirements.txt
# preprocess preferences dataset first 
python src/preprocessing.py
# Stage I: supervised fine-tune 
bash bash/sft.sh
# Stage II&III: EYE and robust alignment 
# DPO series experiments (including DPO, cDPO, rDPO, and them enhanced with PerpCorrect)
bash bash/dpo.sh
# PPO series experiments (including PPO, cPPO, rPPO, and them enhanced with PerpCorrect)
bash bash/ppo.sh
~~~
