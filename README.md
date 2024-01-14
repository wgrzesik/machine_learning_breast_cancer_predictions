## Breast Cancer Early Detection Project

The aim of the project was to develop a model for the classification of mammography images for the recognition of potential cancerous lesions in breast tissue. This classification may help in the early detection of breast cancer.

### Data
The data are from a dataset containing mammography images:
<br>
https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data
<br>
After discarding outlier observations, 569 examples were obtained, which were divided into a learning set (455 examples) and a test set (114 examples).

### Models
The project compared the performance of 3 models:
- Logistic regression
- Decision trees
- Neural networks

### Evaluation
The metrics used for evaluation were accuracy, precision, recall and F1-score. 

### Conclusions
Logistic regression achieved the best F1-score, striking a balance between precision and recovery. When precision is a priority, the neural network model may be more appropriate. In contrast, the decision tree model scored lowest in all metrics, suggesting that it may be less effective for the task at hand.
