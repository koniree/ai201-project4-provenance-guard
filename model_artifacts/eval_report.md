# Trained ML Signal -- Evaluation Report

Dataset: `ai-ga-dataset.csv` (28662 samples, 22929 train / 5733 test, stratified split)


## Logistic Regression (shipped)

- Accuracy: **0.9944**
- Precision: **0.9958**
- Recall: **0.9930**
- F1: **0.9944**
- ROC-AUC: **0.9999**

Confusion matrix (rows=true, cols=predicted, order=[human, ai]):

```
[[2855   12]
 [  20 2846]]
```

```
              precision    recall  f1-score   support

       human       0.99      1.00      0.99      2867
          ai       1.00      0.99      0.99      2866

    accuracy                           0.99      5733
   macro avg       0.99      0.99      0.99      5733
weighted avg       0.99      0.99      0.99      5733

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

## Feature Inspection (what did the model actually learn?)

Top TF-IDF features by prediction direction (Logistic Regression coefficients):

**-> AI-generated:** `this`, `this paper`, `paper`, `this study`, `presents`, `within`, `study`, `through`, `how`, `towards`, `examines`, `findings`, `article`, `such`, `into`

**-> Human-written:** `in the`, `of the`, `and the`, `to the`, `are`, `were`, `is`, `we`, `was`, `for the`, `with the`, `background`, `that the`, `by the`, `however`

Read honestly: some of this is a genuine, well-documented AI-writing tell (GPT-3's habit of framing text as "this paper examines/presents/investigates..." rather than just stating the content). But some of it -- `background`, `however`, `10` -- looks like it's picking up on the CORD-19 corpus's structured-abstract formatting conventions (OBJECTIVE/METHODS/RESULTS/CONCLUSIONS style headers), which is a property of *this dataset*, not of human writing in general. That's a real generalization risk, not a hypothetical one.

## Out-of-Domain Check

The same trained classifier run on the project's actual target domain (short creative/casual text) instead of its training domain (scientific abstracts), using the milestone-4 reference cases:

| Input | P(AI) |
|---|---|
| clearly_ai (AI boilerplate paragraph) | 0.2322 |
| clearly_human (casual ramen review) | 0.0828 |
| borderline_formal_human (academic-style human paragraph) | 0.0601 |
| borderline_edited_ai (lightly-edited AI text) | 0.0387 |

Two problems are visible here: (1) all four probabilities are compressed into a narrow low band (0.04-0.23) compared to the confident 0.01-0.99 spread seen on the training domain, and (2) the ordering is partly wrong -- `borderline_edited_ai` (which IS AI-generated) scores *lower* than `clearly_human`, the opposite of what a working signal should do. This is the direct evidence behind the 0.10 ensemble weight chosen in `scoring.py` -- see README.md "Known Limitations".
