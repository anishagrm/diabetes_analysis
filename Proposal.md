[Project Midterm](./Midterm.md)<br />
[Final Project](./Final.md)<br />
[Project Directory](./README.md)<br />

**Project Proposal: Tanya Pattani, Mitali Bidkar, Anisha Gurram, Srihita Ayinala, Mihir Tupakula**

Proposal Video Link

[Link](https://www.youtube.com/watch?v=GfEYcoWA3UI)

**Individual Contributions**


| Name | Proposal Contributions |
|----------|----------|
| Mitali | Data Preprocessing, Quantitative Metrics| 
| Tanya | Gantt Chart, Introduction / Background| 
| Srihita | ML Algorithms/Models, Project Goals |
| Mihir | Problem Definition |
| Anisha |Github Page, Expected Results |



**Introduction/Background**

We decided our topic to be diabetes prediction based on demographic and patient history. As of 2018, “50% of Americans are the victim of one or more chronic diseases” (Sarwal et. al, 2018) and there are certain people at higher risk such as the characteristics of “obesity and old-age” (Cowie CC et. al, 2018). If not treated in time, diabetes can lead to the detrimental outcomes of “cardiovascular disease, kidney failure…and blindness”. Analyzing big data through statistical and quantitative models can be essential for identifying and predicting this disease in a timely manner while allowing professionals to analyze objective geographical trends. More specifically, our dataset looks at one’s age, gender, hypertension, heart disease history, smoking history, body mass index (BMI), HbA1c_level (average blood sugar levels), and glucose levels to predict if they have diabetes.


Dataset Link

[Link](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)


**Problem Definition**

This project aims to develop a classification model to predict whether a given patient is diabetic or not based on their medical history and demographic information. The prediction model can help in early detection of diabetics, which is crucial to manage the disease, or preventing various complications associated with diabetes and “can decrease risk of  chronic ulceration, infection and amputation” (Aubert et. al, 1995). Healthcare providers can use such models as a part of their decision-making, since they can easily identify at-risk patients and can make adjustments to treatment plans.

**Methods**

Data Preprocessing
- Label encoding for categorical variables
  - The diabetes dataset contains categorical data types (gender, smoking history).
  - ML models require all input data to be numerical – necessary to encode the data. 
  - Label encoding converts each value of a categorical variable into a unique integer. 
- Feature Selection
  - Identify and select the most important features to reduce dimensionality and improve model performance. 
- Feature Scaling
  -  Normalize the data to scale the features to a [0,1] range to ensure that the scale of the data does not bias the model. 

ML Algorithms/Models
- Decision Tree Classifier 
  - Useful for diabetes prediction because they can model nonlinear relationships between features and the target variable. They also make it easier to determine which characteristics are most important in predicting diabetes.
    - Scikit-learn function/class: DecisionTreeClassifier from sklearn.tree
- Random Forest Classifier 
  - Build on decision trees by assembling a set of trees to improve prediction accuracy and control overfitting. This is effective for diabetes prediction because it can capture complex interactions between features while reducing prediction variance, resulting in higher accuracy.
    - from sklearn.ensemble import RandomForestClassifier
- Logistic Regression 
  - Ideal for binary classification problems such as diabetes prediction. It calculates the likelihood that a given set of features belongs to a specific class (diabetic or not).
    - from sklearn.linear_model import LogisticRegression


**Potential Results and Discussion**

Project Goals
 - The goal of this project is to create a predictive model that can accurately classify people as diabetic or non-diabetic based on medical and demographic data, allowing for earlier detection and better diabetes management. The model's effectiveness will be assessed by aiming for an accuracy rate of 90% or higher. A critical focus will be on achieving a recall rate of at least 85% in order to significantly reduce the risk of missing diabetic cases, which is essential for early intervention. The model's overall performance and balance will be evaluated by aiming for an F1 score of 85% or higher, which reflects a harmonious blend of recall and predictive accuracy. 

Quantitative Metrics
  - Accuracy
    - The proportion of predictions that the model classifies correctly
  - Recall
    - The proportion of true positives
    - Preferred because the cost of false negatives is high 
  - F1
    - The harmonic mean of precision and recall, providing a single metric that balances both
    - Preferred because false positives and false negatives are both costly

- Expected Results
  - We would use a confusion Matrix to evaluate model accuracy. This matrix would show the number of cases that were initially positive or negative and their predicted outcome from the model. The confusion matrix should reveal that the number of non-diabetics who were predicted to have diabetes was lower than the number of diabetics predicted to not have diabetes; this indicates the results have a high precision and low recall.

**Project Timeline**

[Link](https://docs.google.com/spreadsheets/d/1gUDTFyGAmQCDtX-jWqNd4cCTpeQH17oV/edit#gid=497782735)

**References**

M. A. Sarwar, N. Kamal, W. Hamid and M. A. Shah, "Prediction of Diabetes Using Machine Learning Algorithms in Healthcare," 2018 24th International Conference on Automation and Computing (ICAC), Newcastle Upon Tyne, UK, 2018, pp. 1-6, doi: 10.23919/IConAC.2018.8748992. keywords: {Machine learning algorithms;Diseases;Prediction algorithms;Machine learning;Diabetes;Big Data;Big data analytics;Predictive Analytics;Machine Learning;Healthcare},

Cowie CC, Casagrande SS, Menke A, Cissell MA, Eberhardt MS, Meigs JB, Gregg EW, Knowler WC, Barrett-Connor E, Becker DJ, Brancati FL, Boyko EJ, Herman WH, Howard BV, Narayan KMV, Rewers M, Fradkin JE, editors. Diabetes in America. 3rd ed. Bethesda (MD): National Institute of Diabetes and Digestive and Kidney Diseases (US); 2018 Aug. PMID: 33651524.

