# EX-05-Feature-Generation


# AIM
To read the given data and perform Feature Generation process and save the data to a file. 

# Explanation
Feature Generation (also known as feature construction, feature extraction or feature engineering) is the process of transforming features into new features that better relate to the target.
 

# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature Generation techniques to all the feature of the data set
### STEP 4
Save the data to the file


# CODE
```
Name: M.D. Harini
Reg No. : 212222230043
```
# Data.csv:
```
import pandas as pd
df=pd.read_csv("data.csv")
df

#feature generation
import category_encoders as ce
be=ce.BinaryEncoder()
ndf=be.fit_transform(df["bin_1"])
df["bin_1"] = be.fit_transform(df["bin_1"])
ndf

ndf2=be.fit_transform(df["bin_2"])
df["bin_2"] = be.fit_transform(df["bin_2"])
ndf2

df1=df.copy()
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder,OneHotEncoder
import category_encoders as ce
be=ce.BinaryEncoder()
ohe=OneHotEncoder(sparse=False)
le=LabelEncoder()
oe=OrdinalEncoder()


df1["City"] = ohe.fit_transform(df1[["City"]])

temp=['Cold','Warm','Hot','Very Hot']
oe1=OrdinalEncoder(categories=[temp])
df1['Ord_1'] = oe1.fit_transform(df1[["Ord_1"]])

edu=['High School','Diploma','Bachelors','Masters','PhD']
oe2=OrdinalEncoder(categories=[edu])
df1['Ord_2']= oe2.fit_transform(df1[["Ord_2"]])
df1

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df2=pd.DataFrame(sc.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df2

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
df3=pd.DataFrame(sc1.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df3

from sklearn.preprocessing import MaxAbsScaler
sc2=MaxAbsScaler()
df4=pd.DataFrame(sc2.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df4

from sklearn.preprocessing import RobustScaler
sc3=RobustScaler()
df5=pd.DataFrame(sc3.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df5
```
# Encoding.csv :
```
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df

#feature generation
import category_encoders as ce
be=ce.BinaryEncoder()
ndf=be.fit_transform(df["bin_1"])
df["bin_1"] = be.fit_transform(df["bin_1"])
ndf

ndf2=be.fit_transform(df["bin_2"])
df["bin_2"] = be.fit_transform(df["bin_2"])
ndf2

df1=df.copy()
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
le=LabelEncoder()
oe=OrdinalEncoder()

df1["nom_0"] = oe.fit_transform(df1[["nom_0"]])
temp=['Cold','Warm','Hot']
oe2=OrdinalEncoder(categories=[temp])
df1['ord_2'] = oe2.fit_transform(df1[['ord_2']])

df1

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df0=pd.DataFrame(sc.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df0

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
df2=pd.DataFrame(sc1.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df2

from sklearn.preprocessing import MaxAbsScaler
sc2=MaxAbsScaler()
df3=pd.DataFrame(sc2.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df3

from sklearn.preprocessing import RobustScaler
sc3=RobustScaler()
df4=pd.DataFrame(sc3.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df4
```
# Titanic.csv :
```
import pandas as pd
df=pd.read_csv("titanic_dataset.csv")
df

#removing unwanted data
df.drop("Name",axis=1,inplace=True)
df.drop("Ticket",axis=1,inplace=True)
df.drop("Cabin",axis=1,inplace=True)

#data cleaning
df.isnull().sum()

df["Age"]=df["Age"].fillna(df["Age"].median())
df["Embarked"]=df["Embarked"].fillna(df["Embarked"].mode()[0])

df.isnull().sum()

df

#feature encoding
from category_encoders import BinaryEncoder
be=BinaryEncoder()
df["Sex"]=be.fit_transform(df[["Sex"]])
ndf=be.fit_transform(df["Sex"])
ndf

df1=df.copy()
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
embark=['S','C','Q']
e1=OrdinalEncoder(categories=[embark])
df1['Embarked'] = e1.fit_transform(df[['Embarked']])
df1

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df2=pd.DataFrame(sc.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df2

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
df3=pd.DataFrame(sc1.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df3

from sklearn.preprocessing import MaxAbsScaler
sc2=MaxAbsScaler()
df4=pd.DataFrame(sc2.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df4

from sklearn.preprocessing import RobustScaler
sc3=RobustScaler()
df5=pd.DataFrame(sc3.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df5
```
# OUPUT:
# Data.csv :
# Initial Dataset:
![image](https://user-images.githubusercontent.com/113497680/232821655-c1c06fe9-aefe-477f-8103-3fc411316ae0.png)
# Binary Encoding:
![image](https://user-images.githubusercontent.com/113497680/232821751-45ffc132-88f6-4c1a-ba8d-8ef3634485f6.png)
![image](https://user-images.githubusercontent.com/113497680/232821818-31cdaa9b-227c-4b9d-94cd-04753d8e8723.png)
# Encoded Dataset:
![image](https://user-images.githubusercontent.com/113497680/232821956-05938865-dcd5-45a6-93ca-63983ad78e04.png)
# Data Scaling using MinMaxScaler:
![image](https://user-images.githubusercontent.com/113497680/232822096-44c0ee1c-3ba8-41e7-bcb3-5e4ca0b169be.png)
# Data Scaling using StandardScaler:
![image](https://user-images.githubusercontent.com/113497680/232822212-7f26f20a-1208-4d23-85b7-7bc81336cf0d.png)
# Data Scaling using MaxAbsScaler:
![image](https://user-images.githubusercontent.com/113497680/232822349-a4532d5a-2b5a-44ab-a05f-36d36a55ccfa.png)
# Data Scaling using RobustScaler:
![image](https://user-images.githubusercontent.com/113497680/232822716-8a912669-8853-4dfd-a257-0efe5dcd4f6f.png)
# Encoding.csv :
# Initial Dataset:
![image](https://user-images.githubusercontent.com/113497680/232822926-ae81ef81-b64b-489e-b475-c1a8bc12a626.png)
# Binary Encoding:
![image](https://user-images.githubusercontent.com/113497680/232823035-e1ec5298-e17d-44bc-8d8b-c3095a5d890c.png)
![image](https://user-images.githubusercontent.com/113497680/232823088-771fbfc5-b5bb-4f09-936d-f8cf2d4294b0.png)
# Encoded Dataset:
![image](https://user-images.githubusercontent.com/113497680/232823259-b722721f-7c0b-40cf-924c-042c63edd1e1.png)
# Data Scaling using MinMaxScaler:
![image](https://user-images.githubusercontent.com/113497680/232823362-1337e495-f0e0-4041-a6b2-9866e3417c10.png)
# Data Scaling using StandardScaler:
![image](https://user-images.githubusercontent.com/113497680/232823546-a3393d28-f57d-4939-bc46-1f2442b7cefa.png)
# Data Scaling using MaxAbsScaler:
![image](https://user-images.githubusercontent.com/113497680/232823756-e411ba27-bab7-4d5f-a2a6-7ffdf07c6fa1.png)
# Data Scaling using RobustScaler:
![image](https://user-images.githubusercontent.com/113497680/232823899-4601a83f-1f1a-42f7-bdeb-966ac4c13635.png)
# Titanic.csv :
# Initial Dataset:
![image](https://user-images.githubusercontent.com/113497680/232824084-94895aaf-338f-462d-8e7c-e8fa8db4a3e2.png)
# Data cleaning before encoding:
![image](https://user-images.githubusercontent.com/113497680/232824290-551f8626-1e24-41ff-8e10-ca7fef128144.png)
# Cleaned Dataset:
![image](https://user-images.githubusercontent.com/113497680/232824453-ee8aeef3-47f0-442e-99bd-29e5f38fe663.png)
# Binary Encoding:
![image](https://user-images.githubusercontent.com/113497680/232824668-2320eb10-7e8d-4595-9db5-2f6f33a359bf.png)
# Encoded Dataset:
![image](https://user-images.githubusercontent.com/113497680/232824829-10f15c3d-924b-40b6-bb75-fa4f035d9edb.png)
# Data Scaling using MinMaxScaler:
![image](https://user-images.githubusercontent.com/113497680/232824999-2d24508f-6bfd-4fa3-a7e5-185bcbd95275.png)
# Data Scaling using StandardScaler:
![image](https://user-images.githubusercontent.com/113497680/232825135-a8cacfa7-7e9b-4e8d-be35-d4125823c9e8.png)
# Data Scaling using MaxAbsScaler:
![image](https://user-images.githubusercontent.com/113497680/232825310-94028f28-37e6-4517-8ce9-a5058c99cbd8.png)
# Data Scaling using RobustScaler:
![image](https://user-images.githubusercontent.com/113497680/232825457-1c99169e-625f-4036-ad6b-40f2b82fedcb.png)
# RESULT:
Feature Generation process and Feature Scaling process is applied to the given data frames sucessfully.

