# Building a supervised learning model to identity credit card in a eCommerce company | Python

Please check the coding file or access via the link below:  
https://colab.research.google.com/drive/1858u49tun3eMF_3m06jWeF4htibCpzxF?usp=sharing  

Author: Nguyễn Hải Long  
Date: 2025-05  
Tools Used: Python  

---

## 📑 Table of Contents  
1. [📌 Background & Overview](#-background--overview)  
2. [📂 Dataset Description & Data Structure](#-dataset-description--data-structure)  
3. [🔎 Final Conclusion & Recommendations](#-final-conclusion--recommendations)

---

## 📌 Background & Overview  

### Objective:
### 📖 This project is about using Python to 

✔️ Build a machine learning model based on the available dataset, with the purpose of detecting fraudulent credit card transactions.   
✔️ Through this process, apply and understand the fundamental steps of developing a machine learning model.    

### 👤 Who is this project for?  

✔️ Decision makers.   
✔️ The company’s information security department.  

---

## 📂 Dataset Description & Data Structure  

### 📌 Data Source  
- Source: Company database.  
- Size: The dataset is 01 csv file.  
- Format: .csv

### 📊 Data Structure & Relationships  
#### 1️⃣ Table used: 
Using the whole dataset.  

#### 2️⃣ Table Schema & Data Snapshot:  
<details>
 <summary>Table using in this project:</summary>

| Field Name | Data Type | Description |
|------------|-----------|-----------|
| (unnamed) | int64 |   |
| (unnamed) | int64 |   |
| trans_date_trans_time | object | The date and time of the transaction. | 
| cc_num | int64 | Credit card number. |
| merchant | object | Merchant who was getting paid. |
| category | object | In what area does that merchant deal. |
| amt | float64 | Amount of money in American Dollars. |
| first | object | First name of the card holder. |
| last | object | Last name of the card holder. |
| gender | object | Gender of the cardholder. |
| street | object | Street of card holder residence. |
| city | object | City of card holder residence. |
| state | object | State of card holder residence. |
| zip | int64 | ZIP code of card holder residence. |
| lat | float64 | Latitude of card holder. |
| long | float64 | Longitude of card holder. |
| city_pop | int64 | Population of the city. |
| job | object | Trade of the card holder |
| dob | object | Date of birth of the card holder. |
| trans_num | object | Transaction ID |
| unix_time | int64 | Unix time which is the time calculated since 1970 to today. |
| merchant_lat | float64 | Latitude of the merchant |
| merchant_long | float64 | Longitude of the merchant |
| is_fraud | int64 | Whether the transaction is fraud(1) or not(0) |

</details>

---

## ⚒️ Main Process  

*Note: Click the white triangle to see codes*  

### 1️⃣ EDA
<details>
 <summary><strong>Import libraries and dataset, copy dataset:</strong></summary>
  
  ```python
  !pip install category_encoders
  
  # import package

  import pandas as pd
  import numpy as np
  from google.colab import drive
  import math
  import category_encoders as ce
  from sklearn.model_selection import train_test_split
  from sklearn.preprocessing import MinMaxScaler
  from sklearn.linear_model import LogisticRegression
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.metrics import balanced_accuracy_score
  from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
  from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
  from sklearn.model_selection import GridSearchCV
  from sklearn.metrics import mean_squared_error, r2_score

  # load dataset

  drive.mount('/content/drive')
  path = '/content/drive/MyDrive/DAC K34/Machine Learning/Mini project 2/mini-project2 .csv'
  
  df = pd.read_csv(path)
  data = df.copy()
  ```
</details>  

#### Understanding data    

<details>
 <summary><strong>Basic data exploration:</strong></summary>

 ```python
 data.head()

 data.info()
 ```

 ![eda_1](https://github.com/longnguyen0102/photo/blob/main/credit_card_fraud-python/eda_1.png)  

 ```python
 data.shape
 ```

 ![eda_2](https://github.com/longnguyen0102/photo/blob/main/credit_card_fraud-python/eda_2.png)

</details>

➡️ The dataframe has 24 columns and 97,748 rows. The first 2 columns are not relevant -> drop them.  

<details>
 
 ```python
 # the first 2 columns are not relevant -> drop them
 data = data.drop(['Unnamed: 0.1','Unnamed: 0'],axis=1)

 data.info()
 ```
 ![eda_3](https://github.com/longnguyen0102/photo/blob/main/credit_card_fraud-python/eda_3.png)

</details>
  
➡️ After dropping, the dataframe now has 22 columns and 97,748 rows.  

<details>
 <summary><strong>Checking unique values:</strong></summary>

 ```python
 # check unique values
 ## print the percentage of unique
 num_unique = data.nunique().sort_values()
 print('---Percentage of unique values (%)---')
 print(100/num_unique)
 ```
</details>

 ![eda_4](https://github.com/longnguyen0102/photo/blob/main/credit_card_fraud-python/eda_4.png)

<details>
 <summary>Checking missing values:</summary>
 
 ```python
 missing_rows_percentage = data.isnull().any(axis=1).mean() * 100
 print(missing_rows_percentage)
 ```

</details>

 ![eda_5](https://github.com/longnguyen0102/photo/blob/main/credit_card_fraud-python/eda_5.png)

<details>
 <summary>Checking duplicates:</summary>
 
 ```python
 duplicate_count = data.duplicated().sum()
 print(duplicate_count)
 ```
 
</details>

![eda_6](https://github.com/longnguyen0102/photo/blob/main/credit_card_fraud-python/eda_6.png)

➡️ As can be seen, 'gender' and 'is_fraud' have the highest percentage of unique values (50%) because they contains only 2 values (M or F in 'gender' and 0 and 1 in 'is_fraud'). There are no missing and duplicated values.  

### 2️⃣ Feature Engineering

<details>
 <summary>Transform as Hour of transaction:</summary>
 
 ```python
 data['tns_hour'] = data['trans_date_trans_time'].apply(lambda x: pd.to_datetime(x, format = '%Y-%m-%d %H:%M:%S').hour)
 data[['trans_date_trans_time','tns_hour']].head()
 ```

</details>

![feature_engineering_1](https://github.com/longnguyen0102/photo/blob/main/credit_card_fraud-python/feature_engineering_1.png)

✅ This code is creating a new column called 'tns_hour' in the data DataFrame. It does this by applying a function to the 'trans_date_trans_time' column. The function converts each date and time string into a datetime object and then extracts the hour from that object. Finally, it displays the first few rows of the original 'trans_date_trans_time' column and the newly created 'tns_hour' column to show the result.  
✅ The purpose of creating the 'tns_hour' column is to extract the transaction hour from the full datetime field ('trans_date_trans_time'). Transaction time can be an important factor in fraud detection, as fraudulent activities may tend to occur during specific hours of the day. By including the 'tns_hour' feature, the model can learn patterns related to the timing of transactions.  

<details>
 <summary>Age of Users:</summary>

 ```python
 data['age'] = (2025 - data['dob'].apply(lambda x: pd.to_datetime(x, format = '%Y-%m-%d').year))
 
 data[['dob','age']].head()
 ```

</details>

![feture_engineering_2](https://github.com/longnguyen0102/photo/blob/main/credit_card_fraud-python/feature_engineering_2.png)

✅ This code creates a new column called 'age' in the data DataFrame. It calculates the age of each person by subtracting the year of their birth (extracted from the 'dob' column after converting it to a datetime object) from the year 2025. Finally, it displays the first few rows of the original 'dob' column and the newly created 'age' column to show the result of the calculation.    
✅ The purpose of creating the 'age' column is to incorporate the user’s age into the model. Age can be an important feature in fraud detection, as different age groups may exhibit different spending behaviors or have varying levels of vulnerability to fraud. By including age as a feature, the model can leverage this information to improve prediction accuracy.  

<details>
 <summary>Distance from user to merchant:</summary>

 ```python
 def harversine(lat1, lon1, lat2, lon2):
   R = 6371  # Earth's radius in km
   lat1, lon1, lat2, lon2 = map(math.radians,[lat1, lon1, lat2, lon2])
 
   dlat = lat2 - lat1
   dlon = lon2 - lon1
   a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2)*math.sin(dlon/2)**2
   c = 2 * math.sin(math.sqrt(a))
 
   return R * c # Distance in km
 
 data['distance'] = data.apply(lambda x: harversine(x['lat'],x['long'],x['merch_lat'],x['merch_long']),axis=1)
 ```

</details>

✅ This code defines a function called 'harversine' that calculates the distance between two points on the Earth given their latitude and longitude using the Haversine formula. It then applies this function to each row of the data DataFrame to calculate the distance between the user's location ('lat', 'long') and the merchant's location ('merch_lat', 'merch_long'), storing the result in a new column called 'distance'. This distance could be a useful feature for fraud detection, as fraudulent transactions might involve unusual distances between the user and the merchant.  
✅ The purpose of creating the 'distance' column is to measure the geographical distance between the user’s location and the merchant’s location. This distance can be an important feature for fraud detection. For example, a transaction occurring unusually far from the user’s typical location may indicate potential fraudulent activity. By calculating and including this distance in the model, we can help the model identify suspicious transactions based on location-related patterns.  

<details>
 <summary>Remove some unused features:</summary>

 ```python
 exclude_cols = ['trans_date_trans_time', 'cc_num','first','last','dob','trans_num','unix_time',
               'long','lat','merch_lat','merch_long']
 data.drop(columns = exclude_cols, inplace=True)

 data.head()
 ```
 
</details>

![feature_engineering_3](https://github.com/longnguyen0102/photo/blob/main/credit_card_fraud-python/feature_engineering_3.png)

➡️ After using all of these features, we drop them so model will learn faster.  

<details>
 <summary>Printing list of columns and numbers of value which have string data type:</summary>

 ```python
 # printing a list of columns and numbers of value which have string data type
 category_cols = data.select_dtypes(include = ['object'])
 for col in category_cols:
  print(f"{col}:{data[col].nunique()}")
 ```
 
</details>

![feature_engineering_4](https://github.com/longnguyen0102/photo/blob/main/credit_card_fraud-python/feature_engineering_4.png)

<details>
 <summary>Encoding 'category', 'gender', 'state' columns: </summary>

 ```python
 # encoding 'category', 'gender', 'state'
 list_columns = ['category','gender','state']
 df_encoded = pd.get_dummies(data, columns = list_columns, drop_first=True)
 ```

 ```python
 # drop all unused columns
 cols = ['merchant','street','city','job']
 df_encoded.drop(columns=cols,inplace=True)
 ```

</details>

➡️ When examining the total number of columns with the data type string, we observe that the columns 'merchant', 'street', 'city', and 'job' contain a very large number of unique values (each over 400). Encoding these columns would create a large number of additional features and significantly increase training time. Since these features are not essential for detecting credit card fraud, we will drop them.  

Instead, we will encode the remaining categorical columns ('category', 'gender', 'state') so that the model can learn more efficiently. In this case, we will convert these columns into boolean (True/False) values.  

### 3️⃣ Model Training  

**In this process, we will do three steps to build a Machine Learning model:**  
**- Split dataset.**  
**- Normalize each dataset.**  
**- Apply model.**  

<details>
 <summary>Split dataset to train, validate, test:</summary>

 ```python
 # drop 'is_fraud' column
 x = df_encoded.drop('is_fraud', axis = 1)
 
 # creating dataframe y with 'is_fraud' column
 y = df_encoded[['is_fraud']]
 
 # split dataset into train (70%) and temp (30%) dataset
 x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3, random_state=42)
 
 # split temp dataset into validate (70%) and test (30%) dataset
 x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.3, random_state=42)
 
 print(f'Number data of train set: {len(x_train)}')
 print(f'Number data of validate set: {len(x_val)}')
 print(f'Number data of test set: {len(x_test)}')
 ```

</details>

![](https://github.com/longnguyen0102/photo/blob/main/credit_card_fraud-python/model_training_1.png)

ℹ️ ***Splitting*** the dataset into training, validation, and test sets serves several critical purposes in machine learning:  
1. *Preventing overfitting:* Overfitting occurs when a model learns patterns too specifically from the training data and fails to generalize to unseen data. By keeping a portion of the dataset (validation and test sets) separate during training, we can evaluate whether the model generalizes well.
2. *Model performance evaluation:* The test set provides an unbiased evaluation of the final model’s performance on completely unseen data. This gives us a realistic estimate of how the model will behave in production or real-world scenarios.  
3. *Model selection and hyperparameter tuning:* The validation set is used during model development to:  
- Compare the performance of different models.  
- Fine-tune hyperparameters to optimize performance without touching the test set.  
In summary, dataset splitting ensures that we build machine learning models that are robust, reliable, and capable of generalizing effectively to new data.  

✅ First, we drop the column is_fraud, since after training the model will generate a new is_fraud column to record the predictions. This allows us to compare the actual values with the model’s outputs. Next, we split the dataset into two parts: a training set (70% of the data) and a temporary set (30%). The temporary set is then further divided into a validation set (70%) and a test set (30%).  

<details>
 <summary>Normalize dataset:</summary>

 ```python
 scaler = MinMaxScaler()

 # fit() method for calculating min and max of each feature in train dataset
 x_train_scaled = scaler.fit_transform(x_train)
 
 x_val_scaled = scaler.transform(x_val)
 
 x_test_scaled = scaler.transform(x_test)
 ```

</details>

ℹ️ ***Normalization*** is a data preprocessing step that rescales the features of a dataset to a standard range. This is often done to improve the performance of machine learning algorithms, as many algorithms are sensitive to the scale of input features.

<details>
 <summary>Apply model:</summary>

 ```python
 # Logistic regression:

 clf_logis = LogisticRegression(random_state = 42)
 clf_logis.fit(x_train_scaled, y_train)
 
 y_pred_val = clf_logis.predict(x_val_scaled)
 y_pred_train = clf_logis.predict(x_train_scaled)
 ```

 ```python
 # Random Forest:
 
 clf_rand = RandomForestClassifier(max_depth=15, random_state=42, n_estimators=100)
 
 clf_rand.fit(x_train_scaled, y_train)
 
 y_ranf_pre_train = clf_rand.predict(x_train_scaled)
 y_ranf_pre_val = clf_rand.predict(x_val_scaled)
 ```

</details>

✅ Applying both approaches to build the machine learning model. The goal is to compare them and identify which approach yields the best evaluation metrics.  

| Logistics Regression | Random Forest |
|----------------------|---------------|
| Logistic Regression is a statistical model used for binary classification problems. In simpler terms, it's used to predict the probability that an observation belongs to one of two categories (like 'yes' or 'no', 'fraudulent' or 'not fraudulent'). | Random Forest is another popular machine learning algorithm, and it's an ensemble method. This means it combines the predictions of multiple individual decision trees to make a final prediction. |
| 1. ***Simplicity and Interpretability:*** Logistic Regression is relatively straightforward to understand and the coefficients of the model can be interpreted as the change in the log-odds of the outcome for a one-unit change in the predictor variable. This makes it easy to explain which factors are influencing the prediction. | 1. ***Improved Accuracy:*** By combining multiple trees, Random Forest often achieves higher accuracy than a single decision tree, especially on complex datasets. It reduces the risk of overfitting that can occur with individual decision trees. |
| 2. ***Efficiency:*** It's computationally less expensive than more complex algorithms, making it suitable for large datasets or situations where quick predictions are needed. | 2. ***Robustness to Outliers:*** Random Forest is less sensitive to outliers in the data compared to some other algorithms. |
| 3. ***Baseline Model:*** It often serves as a good baseline model to compare the performance of more complex algorithms against. | 3. ***Handles Non-linear Relationships:*** It can capture complex, non-linear relationships between features and the target variable. |
|   | 4. ***Feature Importance:*** Random Forest can provide insights into which features are most important for making predictions. |

### 4️⃣ Model Evaluation  

ℹ️ ***Model evaluation*** is a crucial step in the machine learning workflow. It helps us understand how well our trained model performs on unseen data and assess its effectiveness in solving the problem at hand. By evaluating a model, we can:  
1. *Measure Performance:* Quantify the model's accuracy, precision, recall, or other relevant metrics to understand its strengths and weaknesses.
2. *Compare Models:* Determine which model is best suited for the task when comparing different algorithms or hyperparameter settings.  
3. *Identify Overfitting/Underfitting:* Check if the model is too complex (overfitting) or too simple (underfitting) by comparing its performance on training and validation/test datasets.
4. *Tune Hyperparameters:* Use evaluation metrics to guide the process of optimizing the model's hyperparameters for better performance.
5. *Gain Confidence:* Build confidence in the model's ability to generalize to new data before deploying it.  

ℹ️ In this step, using 2 steps for evaluating model (on both Logistic Regression and Random Forest): **calculating balanced accuracy** and **using Confusion Matrix plot for visualization performance**.  
- Calcualting balanced accuracy: this step applies on training and validation data. It compares the model's predictions on the training data to the true values and gives you a balanced measure of how well the model performed on the training set.  
- Confusion matrix: a table that summarizes the performance of a classification model. It shows the number of correct and incorrect predictions made by the model compared to the actual outcomes.  

#### Logistic Regression:  

<details>
 <summary>Balanced accuracy:</summary>

 ```python
 # calculates the balanced accuracy on the training data. Compared the actual target values in training data with prediction on the training data.
 balanced_accuracy_train = balanced_accuracy_score(y_train, y_pred_train)
 
 # calculates the balanced accuracy on the validation data. Compared the actual target values in validation data with prediction on the validation data.
 balanced_accuracy_val = balanced_accuracy_score(y_val, y_pred_val)
 
 print(f'Balanced accuracy training data: {balanced_accuracy_train}')
 print(f'Balanced accuracy validation data: {balanced_accuracy_val}')
 ```
</details>

![](https://github.com/longnguyen0102/photo/blob/main/credit_card_fraud-python/model_training_2.png)

➡️ In this case, the balanced accuracy of approximately 0.69 for both the training and validation sets suggests that the Logistic Regression model has a modest ability to correctly classify both fraudulent and non-fraudulent transactions. A balanced accuracy of 1.0 would indicate perfect performance, while 0.5 would indicate performance no better than random guessing.  
➡️ The close scores between the training and validation sets suggest that the model is not significantly overfitting to the training data. However, a balanced accuracy of 0.69 indicates there is still significant room for improvement in correctly identifying both classes, especially the minority class (fraudulent transactions).  

<details>
 <summary>Confusion matrix:</summary>

 ```python
 cm = confusion_matrix(y_val, y_pred_val, labels=clf_logis.classes_)
 disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf_logis.classes_)
 disp.plot(
 ```
</details>

![](https://github.com/longnguyen0102/photo/blob/main/credit_card_fraud-python/model_training_3.png)

1. **True Negatives (TN):** 18967 - The model is very good at correctly identifying non-fraudulent transactions.  
2. **True Positives (TP):** 570 - The model correctly identifies a significant number of fraudulent transactions.  
3. **False Positives (FP):** 45 - The model incorrectly flags a relatively small number of non-fraudulent transactions as fraudulent.  
4. **False Negatives (FN):** 945 - The model misses a substantial number of fraudulent transactions.

➡️ The Logistic Regression model has a high true negative rate, meaning it's effective at identifying legitimate transactions. However, it has a relatively high number of false negatives, indicating that it is not as effective at catching fraudulent transactions. This suggests that while the model has good precision (when it predicts fraud, it's often correct), it has lower recall (it misses many actual fraudulent cases). In a fraud detection context, minimizing false negatives is often a higher priority than minimizing false positives, as missing fraud is generally more costly than incorrectly flagging a legitimate transaction.  

#### Random Forest:  

<details>
 <summary>Balanced accuracy:</summary>

 ```python
 balanced_accuracy_train = balanced_accuracy_score(y_train,y_ranf_pre_train)
 balanced_accuracy_val = balanced_accuracy_score(y_val, y_ranf_pre_val)
 
 print(f'Balanced accuracy training data: {balanced_accuracy_train}')
 print(f'Balanced accuracy validation data: {balanced_accuracy_val}')
 ```
</details>

![](https://github.com/longnguyen0102/photo/blob/main/credit_card_fraud-python/model_training_4.png)

✅ These balanced accuracy scores (around 0.90) are significantly higher than those of the Logistic Regression model (around 0.69). This indicates that the Random Forest model is much better at correctly classifying both fraudulent and non-fraudulent transactions.  
➡️ Comparing the two models based on balanced accuracy, the Random Forest model is clearly performing much better at identifying both classes accurately.  

<details>
 <summary>Grid Search</summary>

 ```python
 # creating a dictionary to test RandomForestClassifier
 param_grid = {
     'n_estimators': [10,100,200],
     'max_depth' : [None, 15]
 }
 
 grid_search = GridSearchCV(clf_rand, param_grid, cv=5, scoring='balanced_accuracy')
 
 grid_search.fit(x_train, y_train)
 
 print('Best Parameters: ', grid_search.best_params_)
 
 best_clf = grid_search.best_estimator_
 accuracy = best_clf.score(x_test, y_test)
 ```
</details>

✅ The grid search helped you identify the specific settings for your Random Forest model that are likely to give you the best performance on unseen data, according to the balanced accuracy metric.  

<details>
 <summary>Calculating Mean Squared Error & R-squared</summary>

 ```python
 # calculate predictions on the test set using the best estimator
 y_pred_test = best_clf.predict(x_test)
 
 mse = mean_squared_error(y_test, y_pred_test)
 r2 = r2_score(y_test, y_pred_test)
 
 print(f'Mean Squared Error on Test Set: {mse}')
 print(f'R-squared on Test Set: {r2}')
 ```
</details>

![](https://github.com/longnguyen0102/photo/blob/main/credit_card_fraud-python/model_training_5.png)

ℹ️ **Mean Squared Error (MSE)** and **R-squared** are metrics used to evaluate the performance of regression models.  
- *MSE:* Measures the average of the squared errors between the actual values and the model's predicted values. A lower MSE indicates a more accurate model.
- *R-squared:* Measures the goodness of fit of the model to the data. It represents the proportion of the variance in the dependent variable (the variable you want to predict) that is explained by the independent variables (your features). R-squared values range from 0 to 1. A value closer to 1 indicates that the model better explains the data's variability.
➡️ The low MSE (0.01068) suggests that the model's predictions are generally close to the actual values (either 0 or 1). The high R-squared (0.855) indicates that a large proportion of the variability in the 'is_fraud' variable can be explained by the features used in the model. While these are regression metrics, in a classification context with a binary target (0 or 1), they suggest that the model is doing a good job of assigning scores or probabilities that align with the true class labels. This reinforces the findings from the balanced accuracy and confusion matrix that the Random Forest model is performing well.  

<details>
 <summary>Confusion matrix:</summary>

 ```python
 cm_rand = confusion_matrix(y_val, y_ranf_pre_val, labels=clf_rand.classes_)
 disp_rand = ConfusionMatrixDisplay(confusion_matrix=cm_rand, display_labels=clf_rand.classes_)
 disp_rand.plot()
 plt.show()
 ```
</details>

![](https://github.com/longnguyen0102/photo/blob/main/credit_card_fraud-python/model_training_6.png)

1.  **True Negatives (TN):** 18977 - The model correctly identified a large number of non-fraudulent transactions.  
2.  **True Positives (TP):** 1209 - The model correctly identified a significant number of fraudulent transactions. This is a substantial improvement compared to the Logistic Regression model.  
3. **False Positives (FP):** 35 - The model incorrectly flagged a small number of non-fraudulent transactions as fraudulent. This is lower than the Logistic Regression model.  
4. **False Negatives (FN):** 306 - The model missed a number of fraudulent transactions. While lower than the Logistic Regression model, this is still an important area for potential improvement.

➡️ The Random Forest model demonstrates much better performance in identifying fraudulent transactions compared to the Logistic Regression model. It has a higher number of True Positives and a lower number of False Negatives. While it still misses some fraudulent cases, its ability to correctly classify both classes is significantly better, as indicated by the higher balanced accuracy scores previously observed. The relatively low number of False Positives is also a positive sign, suggesting the model is not excessively flagging legitimate transactions as fraudulent. Overall, the Random Forest model appears to be a more suitable choice for this fraud detection task.  

## 📌 Key Takeaways:  
✔️ **Random Forest outperforms Logistic Regression:** The Random Forest model achieved significantly higher balanced accuracy scores (around 0.90) compared to Logistic Regression (around 0.69). This indicates that the Random Forest model is much better at correctly classifying both fraudulent and non-fraudulent transactions.  
✔️ **Random Forest is better at identifying fraud:** The confusion matrix for the Random Forest model shows a much higher number of True Positives (correctly identified fraudulent transactions) and a lower number of False Negatives (missed fraudulent transactions) compared to the Logistic Regression model. This is crucial in a fraud detection scenario where minimizing missed fraud cases is a priority.  
✔️  **Low False Positives in Random Forest:** The Random Forest model also has a relatively low number of False Positives (incorrectly flagged legitimate transactions), which is desirable as it avoids unnecessary disruptions for legitimate users.  
✔️  **Feature Engineering and Model Choice Impact:** The feature engineering steps (like creating 'tns_hour', 'age', and 'distance') and the choice of a more complex model like Random Forest have significantly improved the ability to detect fraudulent transactions compared to a simpler model like Logistic Regression.  
✔️ **Regression Metrics in Classification Context:** While MSE and R-squared are regression metrics, their values in this binary classification context suggest that the Random Forest model's internal scoring or probability assignments align well with the true class labels, further supporting its good performance.  
