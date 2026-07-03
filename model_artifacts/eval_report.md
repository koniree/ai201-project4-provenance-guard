# Trained ML Signal -- Evaluation Report

Dataset: `ai-ga-dataset.csv` (28662 samples, 22929 train / 5733 test, stratified split)


## Logistic Regression (shipped)

- Accuracy: **0.9951**
- Precision: **0.9968**
- Recall: **0.9934**
- F1: **0.9951**
- ROC-AUC: **0.9999**

Confusion matrix (rows=true, cols=predicted, order=[human, ai]):

```
[[2858    9]
 [  19 2847]]
```

```
              precision    recall  f1-score   support

       human       0.99      1.00      1.00      2867
          ai       1.00      0.99      1.00      2866

    accuracy                           1.00      5733
   macro avg       1.00      1.00      1.00      5733
weighted avg       1.00      1.00      1.00      5733

```


## LinearSVC (comparison)

- Accuracy: **0.9969**
- Precision: **0.9983**
- Recall: **0.9955**
- F1: **0.9969**
- ROC-AUC: **1.0000**

Confusion matrix (rows=true, cols=predicted, order=[human, ai]):

```
[[2862    5]
 [  13 2853]]
```

```
              precision    recall  f1-score   support

       human       1.00      1.00      1.00      2867
          ai       1.00      1.00      1.00      2866

    accuracy                           1.00      5733
   macro avg       1.00      1.00      1.00      5733
weighted avg       1.00      1.00      1.00      5733

```
