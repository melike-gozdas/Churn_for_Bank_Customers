import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from collections import Counter
label_encoder = preprocessing.LabelEncoder() 
import warnings
warnings.filterwarnings("ignore")
#from sklearn.neighbors import LocalOutlierFactor
data = pd.read_csv("C:/Users/Melike/Desktop/bitirme_projesi_spyder/churn.csv")
#veri setinin bir kopyasını alma ve öznitelik isimlerini türkçeye çevirme
df_data= data.copy()
df=df_data.copy()
drop_col=['RowNumber', 'CustomerId', 'Surname']
df=df.drop(drop_col,1)
#df_veri.to_csv('df_veri.csv')
df["Geography"] = pd.Categorical(df["Geography"])
df["Gender"] = pd.Categorical(df["Gender"])
#rename columns
mapping = {'HasCrCard':'HasCreditCard'
          }
df=df.rename(columns=mapping)


#ilk 5 satırı yazdırma işlemi
top_five=df.head()

#son 5 satırı yazdırma işlemi
last_five=df.tail()

#veri seti yapısal bilgileri
print("\n\n")
print("Structural information of attributes in the dataset: ")
print(df.info())

#değişkenlere ve değişken tiplerine ulaşma
print("\n\n")
print("Attributes and data types in the dataset: ")
print(df.dtypes)
print("\n\n")

#hiç eksik gözlem(değer) var mı
#değerler üzerinde herhangi birisinde bir tane bile 
#varsa true değeri döner 
print("Are there any missing observations in the dataset?: ")
print(df.isnull().values.any())
print("\n\n")


#Sayısal olarak yüksek değerli bir kategori, düşük değerli bir kategoriden 
#daha yüksek önceliğe sahip olarak kabul edilebilir. O yüzden label encoder kullanılmamıştır.
#df["Geography encoded"]= label_encoder.fit_transform(df["Geography"])
#dfDummies = pd.get_dummies(df['Geography'])
#df=pd.concat([df,dfDummies],axis=1)
#del df["Geography"]

#df["Gender encoded"]= label_encoder.fit_transform(df["Gender"])
#dfDummies1 = pd.get_dummies(df['Gender'])
#df=pd.concat([df,dfDummies1],axis=1)
#del df["Gender"]



print("Unique values found in attributes in the dataset: ")
for i in df.columns:
    if len(df[i].unique())<6:
      print(F'{i}:',len(df[i].unique()),'Values:',df[i].unique())
    else:
      print(F'{i}:',len(df[i].unique()))
print("\n\n")

#sayısal niteliklerin dağılımları
print("Scatter plots of attributes in the dataset: ")
df.hist(figsize = (15,15))
plt.show()
print("\n\n")
#describe işleminin transpozunu olarak değişkenler için yapılan 
#betimsel istatistik daha anlaşılır hale getirme
descriptive_statistics=df.describe().T
print("Descriptive statistics of numeric attributes in the data set: ")
print(descriptive_statistics)
print("\n\n")

numeric_df = list(df.select_dtypes(['int64', 'float64']).columns)
for i in numeric_df:
    sns.distplot(df[i])
    plt.show()    
print("\n\n")
  
categorical_descriptive_statistics=df.describe(include = ['category'])
print(categorical_descriptive_statistics)
print("\n\n")

#Bar plot for gender:
plt.figure(figsize= (3,3))
df['Gender'].value_counts().plot.bar(color= ['m', 'y'])
plt.xlabel('Gender')
plt.ylabel('Count')
plt.xticks(rotation = 0)
plt.show()

print(Counter(df["Gender"]))
print("\n\n")

#bar plot for geography:
plt.figure(figsize = (6, 3))
df['Geography'].value_counts().plot.bar(color = ['m','b','g'])
plt.xlabel('Geography')
plt.ylabel('Count')
plt.xticks(rotation = 0)
plt.show()

#her sınıfın sayısını gösterir
print(Counter(df["Geography"]))
print("\n\n")

