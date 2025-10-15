## EXNO-3-DS
# NAME : NAVEENKUMAR V
# REG NO: 25016071

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
```
<img width="1481" height="549" alt="image" src="https://github.com/user-attachments/assets/d07efe78-28aa-47b9-ad31-c9c35f1a7297" />

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```

 <img width="1482" height="371" alt="image" src="https://github.com/user-attachments/assets/77352375-d0ba-42f0-9af3-c34f9425b092" />

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```

 <img width="1485" height="532" alt="image" src="https://github.com/user-attachments/assets/48593e69-0df7-4410-878b-38328ad17672" />
 
```
le=LabelEncoder()
dfc=df.copy()
df['ord_2']=le.fit_transform(dfc['ord_2'])
df
```

<img width="1479" height="592" alt="image" src="https://github.com/user-attachments/assets/994250ad-146e-4bec-86fd-61d8fce48fcc" />

```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]])) # Orders in Alphabetical Order Blue , Green, Red
df2=pd.concat([df2,enc],axis=1)
df2
```

 <img width="1482" height="644" alt="image" src="https://github.com/user-attachments/assets/d065eaac-0be0-48fa-adcc-73e4261bb119" />
 
```
pd.get_dummies(df2,columns=["nom_0"])
```

 <img width="1446" height="541" alt="image" src="https://github.com/user-attachments/assets/52c68381-6d81-464e-b032-e173cba2c91b" />
 
```
pip install --upgrade category_encoders
```
<img width="1492" height="515" alt="image" src="https://github.com/user-attachments/assets/89b039e9-942c-475e-b21c-87faa0e1699a" />

```
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df
```
<img width="1482" height="615" alt="image" src="https://github.com/user-attachments/assets/b3fcbda7-d71c-44d4-b7b5-a443289e57a4" />

```
be=BinaryEncoder()
d=be.fit_transform(df['Ord_2'])
df=pd.concat([df,nd],axis=1)
df
```
<img width="1491" height="574" alt="image" src="https://github.com/user-attachments/assets/59cf6b34-f309-4a10-a239-16326ac0bff5" />
 
```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
<img width="1481" height="697" alt="image" src="https://github.com/user-attachments/assets/14f3da59-8873-4756-86f3-714b1166d834" />

```
import pandas as pd
from scipy import stats
import numpy as np  
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```
<img width="1488" height="718" alt="image" src="https://github.com/user-attachments/assets/65ef7979-bc5a-4ba2-b6c5-05dd03f20942" />

```
df.skew()
```
 <img width="1488" height="318" alt="image" src="https://github.com/user-attachments/assets/1c58600f-fe96-4f69-99f0-cbb8c8c98e6f" />
 
```
np.log(df["Highly Positive Skew"])
```
<img width="1482" height="631" alt="image" src="https://github.com/user-attachments/assets/3f900d69-e51f-404c-b9f3-9db59696e220" />


```
np.reciprocal(df["Moderate Positive Skew"])
```
 <img width="1482" height="631" alt="Screenshot 2025-10-09 134825" src="https://github.com/user-attachments/assets/91acd498-203f-4314-878f-b88d11e90ca5" />
 
```
np.sqrt(df["Highly Positive Skew"])
```
 <img width="1480" height="635" alt="image" src="https://github.com/user-attachments/assets/4f2449d8-bed7-43f7-affe-8d59fee5bdbf" />
 
```
np.square(df["Highly Positive Skew"])
```
<img width="1489" height="633" alt="image" src="https://github.com/user-attachments/assets/b3813b8f-5a79-4229-a624-2b0c3eb99258" />

```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
 <img width="1484" height="675" alt="image" src="https://github.com/user-attachments/assets/2e03f3e2-e1ad-49a3-833b-1e0c54bd66fd" />
 
```
df.skew()
```
 <img width="1490" height="381" alt="image" src="https://github.com/user-attachments/assets/e4a63a89-870f-44d7-a29b-c65eaa19bf0e" />
 
```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
 <img width="1489" height="423" alt="image" src="https://github.com/user-attachments/assets/2e7a54cc-cbaa-48dc-982f-791a6857248e" />
 
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
<img width="1482" height="681" alt="image" src="https://github.com/user-attachments/assets/e588380c-a503-47b1-afdd-bd5e8f2fb991" />

```
import seaborn as sns
import statsmodels.api as sm # STATS MODEL- STATISTICAL MODEL TO VISUALIZE DISTRIBUTION
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
 <img width="1479" height="717" alt="image" src="https://github.com/user-attachments/assets/572c927b-0ad9-4c23-b4a0-b1d2bbaea6c3" />
 
```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45') # RECIPROCAL
plt.show()
```
 <img width="1489" height="663" alt="image" src="https://github.com/user-attachments/assets/734e6c76-7d3a-4464-99b2-0171f5b74036" />
 
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

<img width="1488" height="718" alt="image" src="https://github.com/user-attachments/assets/525d5cc9-aef1-4d65-81f2-272677be6157" />

# RESULT:

The codes are executed successfully

       
