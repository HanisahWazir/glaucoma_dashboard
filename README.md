# Glaucoma Detection Dashboard

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Put your 5 trained Keras models in a `models/` folder next to `app.py`,
   named:
   ```
   models/resnet50_seed1.keras
   models/resnet50_seed2.keras
   models/resnet50_seed3.keras
   models/resnet50_seed4.keras
   models/resnet50_seed5.keras
   ```
   (You can change the folder path and filename pattern in the sidebar
   instead of renaming files, if you prefer.)

3. Run the app:
   ```
   streamlit run app.py
   ```

## Notes / things to double-check against your actual training pipeline

- **Preprocessing**: the app uses `tensorflow.keras.applications.resnet50.preprocess_input`
  and resizes to 224×224 by default (configurable in the sidebar). If you used
  different preprocessing (e.g. simple `/255.` rescaling, a different input
  size, or manual normalization) during training, update `preprocess_image()`
  in `app.py` to match — otherwise predictions will be wrong.
- **Output format**: the code assumes each model outputs either a single
  sigmoid probability or a 2-unit softmax `[negative_prob, positive_prob]`,
  and takes the last value as "glaucoma positive" probability. If your
  models' output layer is different, adjust `run_ensemble()`.
- **Uncertainty**: the ensemble standard deviation across the 5 models'
  probabilities is used as the uncertainty measure, consistent with your
  Deep Ensemble approach. The cutoff for "low confidence" is a sidebar slider
  you can tune against your validation set.
- The decision threshold defaults to 0.5 but is adjustable in the sidebar —
  useful if you tuned it against sensitivity/specificity during evaluation.
