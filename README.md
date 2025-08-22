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

✔️   
✔️   
✔️   
✔️   
✔️ 

### 👤 Who is this project for?  

✔️   
✔️   
✔️   

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

➡️ This code is creating a new column called tns_hour in the data DataFrame. It does this by applying a function to the trans_date_trans_time column. The function converts each date and time string into a datetime object and then extracts the hour from that object. Finally, it displays the first few rows of the original trans_date_trans_time column and the newly created tns_hour column to show the result.  

<details>
 <summary>Age of Users:</summary>

 ```python
 data['age'] = (2025 - data['dob'].apply(lambda x: pd.to_datetime(x, format = '%Y-%m-%d').year))
 
 data[['dob','age']].head()
 ```

</details>

![feture_engineering_2](https://github.com/longnguyen0102/photo/blob/main/credit_card_fraud-python/feature_engineering_2.png)

➡️  

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

➡️  

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

➡️  

<details>
 <summary>Encoding:</summary>

 ```python
 # printing a list of columns and numbers of value which have string data type
 category_cols = data.select_dtypes(include = ['object'])
 for col in category_cols:
  print(f"{col}:{data[col].nunique()}")
 ```
 
</details>

![feature_engineering_4](https://github.com/longnguyen0102/photo/blob/main/credit_card_fraud-python/feature_engineering_4.png)

➡️ **In this case, we should avoid including fields with too many unique values in the model, as it would significantly increase training time — especially for features such as 'street' or 'city', since we have already calculated the distance from the merchant.**

**The 'job' column, however, can be included as it may improve prediction accuracy. To handle this properly, we should separate 'job' into a dedicated table and link it to the main table through 'id_merchant'. A secondary model can then be built on the 'job' table to group occupations into broader categories (job_category). The main model can subsequently learn using this 'job_category' feature.**

### 3️⃣ 

### 4️⃣ Insights and Actions (drawing from both graphs of RFM and sales trending)  

✔️ The **"Champions"** segment is the core revenue driver: The chart shows that the **"Champions"** group contributes the largest share of revenue—over 60%—despite representing only around 18% of the total customer base. This highlights the critical importance of this segment to SuperStore. These are the most frequent, recent, and high-spending customers.  
➡️ **Action:** It is essential to focus on maintaining and enhancing the experience for **"Champions"** to ensure stable and sustainable revenue.

✔️ The **"Loyal"** segment also makes a significant contribution: The **"Loyal"** customers account for approximately 10% of the total customer base and contribute a notable portion of revenue—over 10%.  
➡️ **Action:** This is a high-potential segment that can be nurtured to become future **"Champions"** Targeted initiatives such as personalized offers, loyalty programs, or incentives could encourage them to increase purchase frequency and order value.  

✔️ The **"Potential Loyalist"** segment shows promise but needs activation: The **"Potential Loyalist"** group represents a relatively high share of the customer base (11%) but contributes only around 3.2% of total revenue. This aligns with the typical characteristics of this segment—good Recency and Frequency, but low Monetary value.  
➡️ **Action:** Targeted campaigns should aim to increase spending per transaction for this group in order to convert them into **"Loyal"** or even **"Champions"** over time. Strategies could include personalized upselling, product bundling, or limited-time promotions to encourage higher basket sizes.

✔️ Based on the *sales trending* graph:  
➡️ Quarter fourth is a good time for **upselling**. This is the time that customers will spend more money for preparing for Holiday Season. Upselling programs are focus on increasing average order value instead of discount.  
➡️ Months in early and middle of the year are the time for launching **customer incentive and relation programs**. During these time, the need for buying is low. That is the reason for these programs to step in, they will attract more customers (even new ones) and increase customers' Frequency, like: price discount, buy 1 get 1, voucher for the next buying,...  
➡️ Months before sales increasing (such as September) is the time for **"heat up the market"**. Launching early promotion programs, new products, new collections are not the bad idea.  

## 📌 Key Takeaways:  
✔️ Understand how **RFM analysis** can be used to evaluate customer behavior based on purchase frequency and spending value.  
✔️ **Classify customers** into specific segments using RFM scores, helping identify which segments require enhanced experiences and which should be retained and nurtured to move toward higher-value tiers.  
✔️ Determine the **optimal timing** for launching promotional campaigns and upselling strategies, enabling the business to both retain existing customers and attract new ones.
