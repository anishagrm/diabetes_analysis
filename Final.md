[Project Proposal](./Proposal.md)<br />
[Project Midterm](./Midterm.md)<br />
[Project Directory](./README.md)<br />

**Project Final Checkpoint: Tanya Pattani, Mitali Bidkar, Anisha Gurram, Srihita Ayinala, Mihir Tupakula**


| Name | Midterm Checkpoint Contributions |
|----------|----------|
| Mitali |Model comparison, next steps, Github| 
| Tanya | Random forest analysis| 
| Srihita | Decision tree analysis |
| Mihir | Decision tree coding |
| Anisha | Random forest coding  |

**Project Timeline (Gantt Chart)**

[Link](https://docs.google.com/spreadsheets/d/1gUDTFyGAmQCDtX-jWqNd4cCTpeQH17oV/edit?usp=sharing&ouid=101906436193615469329&rtpof=true&sd=true)

<h2>Introduction/Background</h2>

Our topic is diabetes prediction using demographic and patient history. In 2018, "50% of Americans" had one or more chronic diseases (Sarwal et al., 2018), with "obesity and old age" being significant risk factors (Cowie CC et al., 2018). Diabetes left untreated leads to "cardiovascular disease, kidney failure…and blindness". Statistical analysis of big data is crucial for disease identification and prediction, aiding in objective geographical trend analysis. Our dataset includes age, gender, hypertension, heart disease, smoking history, BMI, HbA1c_level, and glucose levels for diabetes prediction.


Dataset Link

[Link](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)

Here's some visualizations of our data distribution:

```python
df2 = pd.read_csv('/content/drive/MyDrive/diabetes_prediction_dataset.csv')

plt.figure(figsize=(15, 10))

cont = ['HbA1c_level', 'blood_glucose_level', 'age', 'bmi']
for i, var in enumerate(cont, 1): 
    plt.subplot(3, 3, i)
    sns.histplot(df2[var], kde=True)
    plt.title(f'Distribution of {var}')

cat = ['hypertension', 'smoking_history', 'heart_disease', 'gender']
for i, var in enumerate(cat, len(cont) + 1):
    plt.subplot(3, 3, i)
    sns.countplot(x=var, data=df2)
    plt.title(f'Distribution of {var}')

plt.tight_layout()

plt.show()

```
![image](https://github.gatech.edu/storage/user/68585/files/1b10e373-9c22-40b5-94e2-a347a5215cd4)

<h2>Problem Definition</h2>

This project aims to develop a classification model to predict whether a given patient is diabetic based on their medical history and demographic information. The prediction model can help in the early detection of diabetics, which is crucial to managing the disease, or preventing various complications associated with diabetes and “can decrease risk of chronic ulceration, infection, and amputation” (Aubert et. al, 1995). Healthcare providers can use such models as a part of their decision-making since they can easily identify at-risk patients and can make adjustments to treatment plans.


<h2>Preprocessing</h2>

Our first preprocessing method was label encoding for categorical variables. We did not encode numerical columns such as age, BMI, HbA1c level, blood glucose level. In addition, binary columns such as hypertension, heart disease, and diabetes were already in binary format. Thus, we encoded 'gender' and 'smoking_history'. We used the LabelEncoder from sklearn.preprocessing to perform this task.

| age  | hypertension | heart_disease | bmi   | HbA1c_level | blood_glucose_level | diabetes | gender_encoded | smoking_history_encoded |
|------|--------------|---------------|-------|-------------|---------------------|----------|----------------|-------------------------|
| 80.0 | 0            | 1             | 25.19 | 6.6         | 140                 | 0        | 0              | 4                       |
| 54.0 | 0            | 0             | 27.32 | 6.6         | 80                  | 0        | 0              | 0                       |
| 28.0 | 0            | 0             | 27.32 | 5.7         | 158                 | 0        | 1              | 4                       |
| 36.0 | 0            | 0             | 23.45 | 5.0         | 155                 | 0        | 0              | 1                       |
| 76.0 | 1            | 1             | 20.14 | 4.8         | 155                 | 0        | 1              | 1                       |


We also performed feature scaling, by scaling all the continuous features that had not been encoded already. This was to ensure standardized comparison across all features. We used StandardScaler to perform this.

```python
numerical_features = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level']
```

We are using standardization for the numerical features.

```python
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])
df.head()
```

| age       | hypertension | heart_disease | bmi       | HbA1c_level | blood_glucose_level | diabetes | gender_encoded | smoking_history_encoded |
|-----------|--------------|---------------|-----------|-------------|---------------------|----------|----------------|-------------------------|
| 1.692704  | -0.284439    | 4.936379      | -0.321056 | 1.001706    | 0.047704            | 0        | 0              | 4                       |
| 0.538006  | -0.284439    | -0.202578     | -0.000116 | 1.001706    | -1.426210           | 0        | 0              | 0                       |
| -0.616691 | -0.284439    | -0.202578     | -0.000116 | 0.161108    | 0.489878            | 0        | 1              | 4                       |
| -0.261399 | -0.284439    | -0.202578     | -0.583232 | -0.492690   | 0.416183            | 0        | 0              | 1                       |
| 1.515058  | 3.515687     | 4.936379      | -1.081970 | -0.679490   | 0.416183            | 0        | 1              | 1                       |

Lastly, we used mutual information to perform feature selection. Mutual information is a non-negative value that measures the dependency between the variables. A larger value indicates a stronger relationship with the target variable. The dataset was split into feature and response variables, and we created a dataframe for the results. Then, the data frame was sorted by mutual information in descending order, and the features with non-zero mutual information were selected.

```python
X = df.drop('diabetes', axis=1)
y = df['diabetes']

mutual_info = mutual_info_classif(X, y)
mutual_info_df = pd.DataFrame({'feature': X.columns, 'mutual_info': mutual_info})
mutual_info_sorted = mutual_info_df.sort_values('mutual_info', ascending=False)
mutual_info_sorted
selected_features = mutual_info_sorted[mutual_info_sorted['mutual_info'] > 0]['feature']

df = df[selected_features.tolist() + ['diabetes']]
df.head()
```

| HbA1c_level | blood_glucose_level | age       | bmi      | hypertension  | smoking_history_encoded | heart_disease | gender_encoded | diabetes |
|-------------|---------------------|-----------|-----------|--------------|-------------------------|---------------|----------------|----------|
| 1.001706    | 0.047704            | 1.692704  | -0.321056 | -0.284439    | 4                       | 4.936379      | 0              | 0        |
| 1.001706    | -1.426210           | 0.538006  | -0.000116 | -0.284439    | 0                       | -0.202578     | 0              | 0        |
| 0.161108    | 0.489878            | -0.616691 | -0.000116 | -0.284439    | 4                       | -0.202578     | 1              | 0        |
| -0.492690   | 0.416183            | -0.261399 | -0.583232 | -0.284439    | 1                       | -0.202578     | 0              | 0        |
| -0.679490   | 0.416183            | 1.515058  | -1.081970 | 3.515687     | 1                       | 4.936379      | 1              | 0        |

The DataFrame remained unchanged with all original features plus the diabetes target column. However, with such a small dataset (as the one shown), the reliability of mutual information scores and other feature selection methods may be compromised.

```python 
mutual_info = mutual_info_classif(X, y)
sns.heatmap(pd.DataFrame(mutual_info, index=X.columns, columns=['mutual_info']), cmap="Blues", annot=True)
plt.show()
```
![image](https://github.gatech.edu/storage/user/68585/files/b373bf4f-b5e8-4396-8342-d2a3fa99dfc5)

This plot shows all the mutual information scores for each of the variables, with HbA1c_level having the highest score and gender_encoded having the lowest. This concluded preprocessing.

<h2>Logistic Regression Model</h2>

Logistic regression was chosen since it works well for binary classification tasks. It simulates the likelihood that an input falls into a specific category (in this case, diabetes or not). We used LogisticRegression from sklearn to build the model. A 20-80 split was used to split the data into training and testing sets, with 20% of the data as testing data and 80% as training data. We also used sklearn’s inbuilt classification_report to report the precision, recall, F1, and support scores for the model. These will be expanded on later.

```python 
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.5f}\n")
conf_matrix_df = pd.DataFrame(confusion_matrix(y_test, y_pred), index=['Actual Negative', 'Actual Positive'], columns=['Predicted Negative', 'Predicted Positive'])
print("Confusion Matrix:")
print(conf_matrix_df)
print("\n")
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
print("Classification Report:")
print(report_df)
```
Accuracy: 0.95865

Confusion Matrix:

|                 | Predicted Negative | Predicted Positive |
|-----------------|--------------------|--------------------|
| Actual Negative | 18127              | 165                |
| Actual Positive | 662                | 1046               |

Classification Report:

|               | precision | recall  | f1-score | support  |
|---------------|-----------|---------|----------|----------|
| 0             | 0.964767  | 0.990980 | 0.977697 | 18292.0000 |
| 1             | 0.863749  | 0.612412 | 0.716684 | 1708.0000  |
| accuracy      |           |         | 0.958650 | 0.95865   |
| macro avg     | 0.914258  | 0.801696 | 0.847191 | 20000.0000 |
| weighted avg  | 0.956140  | 0.958650 | 0.955407 | 20000.0000 |


<h2>Logistic Regression Results and Discussion</h2>

```python 
plt.figure(figsize=(5, 3.5))
sns.heatmap(conf_matrix_df, annot=True, fmt="d", cmap="Blues", linewidths=.5)
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()
```
![image](https://github.gatech.edu/storage/user/68585/files/afc7995e-795c-4fac-b248-be3282a34519)

Given the values from the confusion matrix:

True Negative (TN): 18127 \
False Positive (FP): 165 \
False Negative (FN): 662 \
True Positive (TP): 1046 

The formulas for the quantative metrics are:

Accuracy = (TP + TN) / (TP + TN + FP + FN) \
Recall = TP / (TP + FN) \
Precision = TP / (TP + FP) \
F1 = 2 * (Precision * Recall) / (Precision + Recall)

Evaluating the formulas with the above values we get:

Accuracy = (1046 + 18127) / (1046 + 18127 + 165 + 662) ≈ 95.865% \
Recall = 1046 / (1046 + 662) ≈ 61.241% \
F1 Score = 2 * ((1046 / (1046 + 165)) * (1046 / (1046 + 662))) / ((1046 / (1046 + 165)) + (1046 / (1046 + 662))) ≈ 71.668% 

```python 
performance_metrics_df = report_df.drop('support', axis=1)

plt.figure(figsize=(5, 3))
sns.heatmap(performance_metrics_df, annot=True, cmap="Blues", linewidths=.5, fmt=".2f")
plt.title('Classification Report')
plt.show()
```

We also obtained this classification report:

![image](https://github.gatech.edu/storage/user/68585/files/67204c78-5251-48f1-83fb-ddfba3fb7300)

The report for each class (0 for the negative class which represents non-diabetes cases, and 1 for the positive class which likely represents diabetes cases) shows precision, recall, and f1-score.

**Precision**: This is the ratio of correctly predicted positive cases to the total predicted positive cases. So essentially, a high precision implies a low false positive rate. For the non-diabetes class (0), the precision is approximately 96.47%, which is relatively high. This suggests that the model is reliable in predicting cases of non-diabetes. For the diabetes class (1), the precision is approximately 86.37%, which is not as precise as the non-diabetes class, but still reasonably high. This indicates that most of the cases predicted as diabetes are actually diabetes. \
**Recall**: This is the ratio of correctly predicted positive cases to all cases in the class, also known as the true positive rate. The non-diabetes (0) class has a recall of approximately 99.90%, which shows that the model is very good at predicting non-diabetes cases. But on the other hand, the diabetes class (1) has a lower recall of about 61.24%. This suggests that the model misses a good number of actual diabetes cases. \
**F1-Score:** This is the harmonic mean of precision and recall. So, it assesses both false positives and false negatives and shows if a class has a balance of precision and recall. The F1 score for the non-diabetes class (0) is approximately 97.77%, while for the diabetes class (1) it's lower at 71.66%. This suggests that there is not a good balance between precision and recall for the diabetes class. \
**Accuracy**: This is the ratio of correctly predicted cases to the total number of cases. The overall accuracy is 95.86%, which implies a high-performing model. \
**Macro Average**: This averages the unweighted mean per label, so it doesn't take label imbalance into account. The macro average for precision, recall, and F1-score are approximately 91.42%, 80.16%, and 84.72% respectively. This suggests the model performs well overall, but there is an imbalance in performance between the two classes. \
**Weighted Average**: This averages the support-weighted mean per label. This accounts for label imbalance by weighing the metrics by the number of true instances for each label. Support is the number of cases of a class in the dataset. The weighted averages for precision, recall, and F1-score are approximately 95.61%, 95.86%, and 95.54% respectively, which suggests that when noting the imbalance, the model still demonstrates a high performance.

In the context of predicting diabetes, the relatively low recall for diabetes cases (1) is something that needs to improve, because it suggests that the model fails to predict a decent portion of actual positive diabetes cases. This is an important issue especially considering this is used in the context of medical diagnosis as failing to diagnose a disease could lead to serious health problems experienced by the patient.

**Logistic Regression Model Improvements**

A goal to improve this model would be to address its low recall. There are multiple different approaches that can be taken to mitigate this issue. We can try lowering the threshold of 0.5 in the logistic regression model. However, this might lead to more false positives. We could also implement more cost-sensitive training through the class_weight parameter. We can adjust this parameter to penalize when there's a misclassification of the positive class more than the negative class. This will incentivize the model to reduce the number of false negatives and thereby increase recall. Finally, we could employ more feature training to have better distinctions between the two classes. By denoising our model and removing irrelevant features, the model can become better at identifying true positive cases (which it was lacking in previously).

<h2>Decision Tree Model</h2>
  
We used a Decision Tree Classifier, a non-parametric supervised learning technique, to solve our diabetes prediction problem. By learning simple decision rules inferred from the data features, the classifier creates a model that forecasts the value of the target variable.
For reproducibility, we used the DecisionTreeClassifier from sklearn.tree with a specific random_state. All columns other than the final one, which represented the target variable y, were used to derive our feature matrix X.
Our dataset was split into a 20–80 test–train ratio, with 20% set aside for model testing. The X_train and y_train datasets were used to train the classifier.

```python

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.5f}\n")

conf_matrix_df = pd.DataFrame(
    confusion_matrix(y_test, y_pred),
    index=['Actual Negative', 'Actual Positive'],
    columns=['Predicted Negative', 'Predicted Positive']
)
print("Confusion Matrix:")
print(conf_matrix_df)
print("\n")

report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
print("Classification Report:")
print(report_df)
``` 
**Model Evaluation**

Accuracy - The Decision Tree Classifier achieved an accuracy of 95.23%, which indicates a high level of overall correct predictions. 

```python
plt.figure(figsize=(5, 3.5))
sns.heatmap(conf_matrix_df, annot=True, fmt="d", cmap="Blues", linewidths=.5)
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()
```

![image](https://github.gatech.edu/storage/user/71766/files/e079ed0f-4b5e-4f6d-8738-ada26a9fb62c)

The Confusion Matrix has revealed the following:
True Negative (TN): 17786, meaning the model correctly predicted the non-diabetic cases
False Positive (FP): 506, where the model incorrectly predicted diabetes
False Negative(FN): 448, where the model failed to correctly identify diabetic causes
True Positive (TN): 1260, representing correct diabetic predictions

These values suggest that even though the model is highly useful and robust in predicting non-diabetic cases, it has room for improvement in correctly identifying diabetic cases, as seen by the number of false negatives. 

Classification report:

```python
performance_metrics_df = report_df.drop('support', axis=1)

plt.figure(figsize=(5, 3))
sns.heatmap(performance_metrics_df, annot=True, cmap="Blues", linewidths=.5, fmt=".2f")
plt.title('Classification Report')
plt.show()
```

![image](https://github.gatech.edu/storage/user/71766/files/97c60fa5-ce9c-4671-9dcf-d064ee99ec64)

- The precision is almost 97.54% for the non-diabetic class (0) and 71.34% for the diabetes class (1). In comparison to instances with diabetes, the model predicts non-diabetic cases more accurately.
- The recall rate is considerably lower for the diabetic class (73.77%) than it is for the non-diabetic class (97.23%). This suggests that the model's capacity to identify actual instances of diabetes has to be strengthened.
- Precision and recall are balanced by the F1-score, which is higher for the non-diabetic class (about 97.38%) and lower for the diabetic class (72.54%).
- Accuracy (0.95): 95% of the time, the Decision Tree Classifier correctly predicted both the diabetes and non-diabetic classifications. A high accuracy rate indicates that the model can consistently distinguish between the classes over the full dataset.
- Macro Average (Precision: 0.84, Recall: 0.86, F1-Score: 0.85): The model's average performance over the two classes is displayed by the macro average, which offers a performance estimate that treats every class equally. It provides equal weight to the performance measurements of the majority and minority classes since it ignores class inequality. The macro average scores, which are close to 0.85, indicate that the model performs reasonably well in both classes, however they might not accurately represent results from an unbalanced dataset.
- Weighted Average (Precision: 0.95, Recall: 0.95, F1-Score: 0.95): By dividing the number of true occurrences in each class by the performance indicator, the weighted average accounts for the imbalance in class sizes. Because it takes into consideration the relative sizes of each class, this provides a more realistic assessment of the model's predictive ability on the dataset. The model is quite effective at predicting in proportion to how common each class is in the data, as seen by the high weighted average scores.

Here are the calculations for the metrics:

![image](https://github.gatech.edu/storage/user/71766/files/33ceaaad-aac2-4d25-8155-06ed5d978d87)

**Decision tree model improvements**

The Decision Tree Classifier is highly accurate in predicting diabetes, with strong precision and recall in non-diabetic situations. However, the recall for diabetes cases can be improved to ensure that the model does not overlook genuine diabetic patients.

Given the model's performance metrics here's some improvements:
Pruning the Tree: To prevent overfitting, we could prune the decision tree to limit its depth or the minimum number of samples required at a leaf node.
Feature Engineering: Exploring more features or creating new ones could help the model differentiate better between diabetic and non-diabetic cases.
Ensemble Methods: Employing ensemble methods like Random Forests could improve performance by overcoming the limitations of a single decision tree.

The Decision Tree Classifier is a reliable model for making initial diabetes predictions. Its transparency and interpretability make it an appealing choice for healthcare professionals. Yet additional modification is required to improve its clinical value, particularly in reducing the number of missed diabetes cases.

<h2>Random Forest Model</h2>

Another model we decided to implement was the Random Forest classifier for its strength in accuracy and ability to capture complex features interactions and predicting variance. Since we have both numerical and categorical variables in our dataset ranging from gender to age to blood glucose levels. With this model, these complex relationships can be handled and overfitting is avoided due to its use of bootstrapping to get multiple training sets and its use of feature randomizing. It can also rank the importance of different features which is especially helpful in figuring out which predictors impact diabetes diagnosis. We began by using the sklearn library to create a model for the random forest classifier which splits the data into an 80-20 training set allowing the model to be trained on separate datasets. The model is built with 100 trees and is then traited.

```python
X = df.drop('diabetes', axis=1)
y = df['diabetes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.5f}\n")
conf_matrix_df = pd.DataFrame(confusion_matrix(y_test, y_pred), index=['Actual Negative', 'Actual Positive'], columns=['Predicted Negative', 'Predicted Positive'])
print("Confusion Matrix:")
print(conf_matrix_df)
print("\n")
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
print("Classification Report:")
print(report_df)
```

![image](https://github.gatech.edu/storage/user/71766/files/5d45fafe-09ce-4080-b6aa-e3d338357d03)

Given the values from the confusion matrix:

True Negative (TN): 18230
False Positive (FP): 62
False Negative (FN):528
True Positive (TP): 1180

Given these values, we can attain some performance metrics.

For accuracy, we attain the value of 0.9705 using this formula below:

![image](https://github.gatech.edu/storage/user/71766/files/f17b4ed8-ed01-4c63-af95-5454b0ee777b)

For precision, we attain the value of 0.95 using this formula below:

![image](https://github.gatech.edu/storage/user/71766/files/3e8950bf-3907-4dbb-986a-e0bbfa93c7b7)

For recall, we attain the value of 0.69 using this formula below:

![image](https://github.gatech.edu/storage/user/71766/files/5491e3fc-37b1-4022-a13f-85ca14bc8e3f)

Together, with these values, we can calculate the F1 score which gives us a way to measure a blend of precision and recall.

![image](https://github.gatech.edu/storage/user/71766/files/67ea4fb6-6164-4bac-9f01-b2c1326ae300)

This gives us an F1 score of 0.8, which is 80%.

We can further analyze these results by looking at the Classification Report below:

```python
performance_metrics_df = report_df.drop('support', axis=1)

plt.figure(figsize=(5, 3))
sns.heatmap(performance_metrics_df, annot=True, cmap="Blues", linewidths=.5, fmt=".2f")
plt.title('Classification Report')
plt.show()
```

![image](https://github.gatech.edu/storage/user/71766/files/10bcc398-1cb1-4dbe-ad1d-490080b7f0a6)

- Precision: For class 0, which is the non-diabetic class, the precision is 0.97 which indicates that the model accurately predicts no diabetes 97% of the time. On the other hand, for class 1, which is predicting diabetes, the model accurately does that 95% of the time. 
- Recall: For class 0, the recall of 1.00 indicates that the model is perfect in identifying all cases of ‘no diabetes’ correctly, which is excellent. For class 1, the recall is only 0.69, which is relatively lower, meaning that only 69% of the diabetes cases were identified correctly.
- F1-Score: For class 0, the F1 score is 0.98 which is very high, showcasing the excellent balance between precision and recall. As we calculated before, for class 1, the F1 score for predicting diabetes is 0.80 which highlights that there is a reasonable balance between precision and recall; however, it definitely could be better.
- Accuracy: As shown above from the printed statements, this model has an overall accuracy of 97% which is great and very high. It allows us to accurately predict if one has diabetes 97% of the time.
- Macro Average: This classification allows us to calculate the metrics (precision, recall, F1) for each class independently and then take the average of it without taking into account the frequency of it. From the report, we can see the macro average precision is 0.94, the macro average recall is 0.84, and the macro average F1-score is 0.89. The F1-score accurately depicts that while the precision score is high, the recall, which shows the ratio of correctly predicted cases, is lower and has room for improvement for identifying positive instances.
- Weighted Average: This is another way to identify the average performance of a model across all the classes while taking into account the imbalance in the distribution of the classes, giving more weight to the larger class. We can see from the report that the weighted average precision is 0.97, the weighted average recall is 0.97, and thus the F1 score is also 0.97. This shows that the performance for class 0, which has the most instances, is very strong and influences the overall metrics.

**Random Forest model improvements**

Overall, from our analysis we can see that our model is better at identifying the negative classes. This is displayed by the lower number of positive recalls, and thus, even though the model is overall accurate, it needs to be improved in correctly identifying positive cases of diabetes. If an actual case of diabetes is missed in a diagnosis, it could have severe implications for a patient.

To address the low recall for class 1, we could apply some resampling techniques to fix the class imbalances. This will help balance the dataset and improve performance for the underrepresented class. In addition, we could also adjust the class_weight parameter such that we could penalize mistakes on the minority class (1) to the amount of how proportionally underrepresented it is. Finally, we could address feature selection and add that preprocessing method to select the most relevant features; this would help us focus on the most important predictors.

<h2>Model Comparison</h2>

**Logistic regression**

Strengths: Logistic regression is effective for binary classification problems like diabetes prediction, as it models the probability of a binary response based on one or more predictor variables. It is relatively simple and interpretable, making it easy to understand the impact of each variable.

Limitations: Logistic regression can struggle with non-linear relationships unless transformations or interactions are included. It also assumes linearity between the log-odds of the outcome and each predictor.

Performance: The logistic regression model showed high overall accuracy (95.86%) and performed particularly well in identifying non-diabetic cases with a recall of 99.90% for class 0. However, its ability to detect diabetic cases (class 1) was weaker, with a recall of only 61.24%, indicating many diabetic cases were missed.

**Decision Tree**

Strengths: Decision trees are non-parametric and can model complex relationships through hierarchical decision rules. They are highly interpretable, as one can follow the path in the tree to understand the decision-making process.

Limitations: Prone to overfitting, especially with many features or deep trees. They can also be unstable, as small changes in the data might result in a very different tree being generated.

Performance: The decision tree achieved an accuracy of 95.23%. It had a precision of 71.34% for diabetic cases, indicating less reliability in predicting true diabetic cases compared to logistic regression. Its recall for diabetic cases was slightly higher than logistic regression at 73.77%, but still indicated room for improvement.

**Random Forest**

Strengths: Random forests mitigate some of the overfitting issues seen with decision trees by ensemble learning, where multiple trees are generated and their results averaged. This model can handle a mix of feature types well and is robust against overfitting compared to single decision trees.

Limitations: Random forests are less interpretable than single decision trees because they involve multiple trees. They can also be computationally intensive, especially with a large number of trees or deep trees.

Performance: This model showed the highest overall accuracy (97.05%) and a significant improvement in handling diabetic cases with a recall of 69% and precision of 95%. This suggests a better balance between identifying diabetic and non-diabetic cases than the decision tree and logistic regression models.

**Model comparison and tradeoffs**

Accuracy and Recall Tradeoffs: Logistic regression and decision trees showed similar accuracies but struggled with recall for diabetic cases. Random forest offered the best balance, achieving higher accuracy and better recall for diabetic cases.

Interpretability versus Accuracy: Logistic regression and decision trees offer higher interpretability, which is crucial for clinical settings where understanding the decision rationale is important. Random forests, while less interpretable, provided better accuracy and managed class imbalance more effectively.

Computational Efficiency: Logistic regression is generally faster and less computationally intensive than both decision tree and random forest, which is an important consideration for deployment in real-time systems or on limited-resource platforms.



Each model has its strengths and appropriate use cases depending on the priority between accuracy, interpretability, and computational efficiency. Random forest emerges as the most robust model in this scenario, offering the best overall performance with regards to handling both classes effectively, albeit at the cost of reduced interpretability and increased computational demands. For clinical applications, enhancing model interpretability while maintaining high accuracy, as seen in the random forest, could be vital for user trust and understanding.

<h2>Next Steps</h2>

**Feature engineering and selection**
- Exploring additional features or creating new ones that might improve the models' predictive power.
- Utilizing more advanced feature selection techniques, such as recursive feature elimination or feature importance ranking, to identify the most relevant predictors for diabetes prediction.

**Model tuning and optimization**
- Fine-tuning hyperparameters for each model to optimize their performance further. For example, we can adjust regularization parameters for logistic regression, max depth or minimum samples per leaf for decision trees, and number of trees or maximum features for random forests.
- Experimenting with different splitting criteria and pruning strategies for decision trees to prevent overfitting and improve generalization.

**Handling imbalance**
- Addressing the class imbalance issue, especially for diabetic cases, by employing resampling techniques like oversampling (e.g., SMOTE) or undersampling to balance the dataset. This could help improve recall for the minority class.
- Adjusting class weights or using other techniques within the models (e.g., class_weight parameter in sklearn) to penalize misclassifications of the minority class more heavily.

**Ensemble methods**
- Exploring other ensemble methods beyond random forests, such as gradient boosting machines (e.g., XGBoost, LightGBM), which may offer even better performance by sequentially building weak learners and correcting errors made by previous models.


<h2>References</h2>
M. A. Sarwar, N. Kamal, W. Hamid and M. A. Shah, "Prediction of Diabetes Using Machine Learning Algorithms in Healthcare," 2018 24th International Conference on Automation and Computing (ICAC), Newcastle Upon Tyne, UK, 2018, pp. 1-6, doi: 10.23919/IConAC.2018.8748992. keywords: {Machine learning algorithms;Diseases;Prediction algorithms;Machine learning;Diabetes;Big Data;Big data analytics;Predictive Analytics;Machine Learning;Healthcare},
<br></br>
Cowie CC, Casagrande SS, Menke A, Cissell MA, Eberhardt MS, Meigs JB, Gregg EW, Knowler WC, Barrett-Connor E, Becker DJ, Brancati FL, Boyko EJ, Herman WH, Howard BV, Narayan KMV, Rewers M, Fradkin JE, editors. Diabetes in America. 3rd ed. Bethesda (MD): National Institute of Diabetes and Digestive and Kidney Diseases (US); 2018 Aug. PMID: 33651524.
<br></br>
Aubert, Ronald E. “Diabetes in America, 2nd Edition.” National Institute of Diabetes and Digestive and Kidney Diseases, National Institutes of Health, Jan. 1995, www.niddk.nih.gov/about-niddk/strategic-plans-reports/diabetes-in-america-2nd-edition. 
