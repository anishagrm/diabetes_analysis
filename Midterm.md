[Project Proposal](./Proposal.md)<br />
[Final Project](./Final.md)<br />
[Project Directory](./README.md)<br />

**Project Midterm Checkpoint: Tanya Pattani, Mitali Bidkar, Anisha Gurram, Srihita Ayinala, Mihir Tupakula**


| Name | Midterm Checkpoint Contributions |
|----------|----------|
| Mitali |Feature selection, Report, GitHub| 
| Tanya | Introduction / Background, Label encoding| 
| Srihita | Logistic regression model |
| Mihir | Feature scaling |
| Anisha | Quantitative metrics, confusion matrix, classification report, analysis, GitHub  |

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


<h2>Results and Discussion</h2>

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

**Next Steps**

A goal to improve this model would be to address its low recall. There are multiple different approaches that can be taken to mitigate this issue. We can try lowering the threshold of 0.5 in the logistic regression model. However, this might lead to more false positives. We could also implement more cost-sensitive training through the class_weight parameter. We can adjust this parameter to penalize when there's a misclassification of the positive class more than the negative class. This will incentivize the model to reduce the number of false negatives and thereby increase recall. Finally, we could employ more feature training to have better distinctions between the two classes. By denoising our model and removing irrelevant features, the model can become better at identifying true positive cases (which it was lacking in previously).


<h2>References</h2>
M. A. Sarwar, N. Kamal, W. Hamid and M. A. Shah, "Prediction of Diabetes Using Machine Learning Algorithms in Healthcare," 2018 24th International Conference on Automation and Computing (ICAC), Newcastle Upon Tyne, UK, 2018, pp. 1-6, doi: 10.23919/IConAC.2018.8748992. keywords: {Machine learning algorithms;Diseases;Prediction algorithms;Machine learning;Diabetes;Big Data;Big data analytics;Predictive Analytics;Machine Learning;Healthcare},
<br></br>
Cowie CC, Casagrande SS, Menke A, Cissell MA, Eberhardt MS, Meigs JB, Gregg EW, Knowler WC, Barrett-Connor E, Becker DJ, Brancati FL, Boyko EJ, Herman WH, Howard BV, Narayan KMV, Rewers M, Fradkin JE, editors. Diabetes in America. 3rd ed. Bethesda (MD): National Institute of Diabetes and Digestive and Kidney Diseases (US); 2018 Aug. PMID: 33651524.
<br></br>
Aubert, Ronald E. “Diabetes in America, 2nd Edition.” National Institute of Diabetes and Digestive and Kidney Diseases, National Institutes of Health, Jan. 1995, www.niddk.nih.gov/about-niddk/strategic-plans-reports/diabetes-in-america-2nd-edition. 




