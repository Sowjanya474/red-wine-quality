# Red Wine Quality Prediction

Machine learning project that predicts whether a red wine is rated **good** (quality 7 or above) from 11 physicochemical measurements, with an interactive Streamlit app.

**Live demo:** https://red-wine-quality-2bgrxsapqvvizh4yc9tuzr.streamlit.app/

**Website:** https://sowjanya474.github.io/red-wine-quality/

**Presentation:** [Download the slides](docs/Red_Wine_Quality.pptx)

**Report:** [Read the project report](docs/Red_Wine_Quality_Report.pdf)

## Dataset
UCI Wine Quality (red), Cortez et al. (2009): 1,599 samples, 11 features, quality scores 3 to 8. Licensed CC BY 4.0.
https://archive.ics.uci.edu/dataset/186/wine+quality

## Approach
1. Removed 240 duplicate rows (1,359 remain) to avoid train/test leakage.
2. Defined the target: good = quality >= 7 (13.5% of wines, so the data is imbalanced).
3. Stratified 80/20 train/test split, with scaling inside a pipeline.
4. Compared Logistic Regression, Decision Tree, Random Forest, SVM and Gradient Boosting using 5-fold cross-validation.
5. Tuned the top models with GridSearchCV (F1 score).
6. Chose the decision threshold (0.37) using training data only.
7. Evaluated once on the held-out test set and explained the model with feature importance and SHAP.

## Results (test set, 272 wines)
| Metric (good wines) | Value |
|---|---|
| Recall | 0.76 |
| Precision | 0.41 |
| F1 | 0.53 |
| ROC-AUC | 0.88 |

Cross-validated F1 for the tuned Random Forest, SVM and Logistic Regression was 0.521, 0.517 and 0.509, so the models performed comparably.

## Key findings
Alcohol, sulphates and volatile acidity were the most influential features. Higher alcohol and sulphates and lower volatile acidity increased the predicted chance of a good rating. These are associations in this dataset, not proof of cause.

## Limitations
- Quality scores come from human tasters, so they are noisy.
- Only 37 good wines in the test set, so the metrics are approximate.
- Trained on Portuguese Vinho Verde wines only.

## Run locally
    pip install -r requirements.txt
    streamlit run app.py