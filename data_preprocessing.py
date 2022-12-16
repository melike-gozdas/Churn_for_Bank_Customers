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


print("Loss segmentation by gender: ")
# Cinsiyete göre "Çıkış" bölümünü ayırın ve her sınıf içindeki sıklığı ve yüzdeyi görüntüleyin:
gender_grouped = df.groupby('Gender')['Exited'].agg(Count = 'value_counts')
print(gender_grouped)
print("\n\n")
    
gender_dfgc = gender_grouped
gender_dfgc = gender_dfgc.pivot_table(values = 'Count', index = 'Gender', columns = ['Exited'])
print(gender_dfgc) 
print("\n\n")

#Cinsiyete göre "Exited" bölümünü ayırıp her sınıf içindeki sıklık yüzdesini görüntüleme:
gender_dfgp = gender_grouped.groupby(level=[0]).apply(lambda i: round(i * 100 / i.sum(), 2))
gender_dfgp.rename(columns={'Count': 'Percentage'}, inplace=True)
print(gender_dfgp)
print("\n\n")

# Yüzdelik tablosunun düzenlenmiş hali
gender_dfgp = gender_dfgp.pivot_table(values = 'Percentage', index = 'Gender', columns = ['Exited'])
print(gender_dfgp)
print("\n\n")


# Cinsiyete göre kayıp dağılımı:

labels= ['Stays', 'Exits']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

gender_dfgc.plot(kind='bar',
          color=['g', 'r'],
          rot=0, 
          ax=ax1)
ax1.legend(labels)
ax1.set_title('Churn Risk per Gender- Count', fontsize=14, pad=10)
ax1.set_ylabel('Count',size=12)
ax1.set_xlabel('Gender', size=12)


gender_dfgp.plot(kind='bar',
          color=['g', 'r'],
          rot=0, 
          ax=ax2)
ax2.legend(labels)
ax2.set_title('Churn Risk per Gender- Percentage', fontsize=14, pad=10)
ax2.set_ylabel('Percentage',size=12)
ax2.set_xlabel('Gender', size=12)

plt.show()
print("\n\n")


print("Loss segmentation by geography: ")
#Coğrafyaya göre "Exited" bölümünü ayırıp her sınıf içindeki sıklığı görüntüleme:
geography_grouped = df.groupby('Geography')['Exited'].agg(Count = 'value_counts')
print(geography_grouped)
print("\n\n")

geography_dfgc =geography_grouped
geography_dfgc = geography_dfgc.pivot_table(values = 'Count', index = 'Geography', columns = ['Exited'])
print(geography_dfgc)
print("\n\n")

#Coğrafyaya göre "Exited" bölümünü ayırıp her sınıf içindeki sıklık yüzdesini görüntüleme:

geography_dfgp = geography_grouped.groupby(level=[0]).apply(lambda i: round(i * 100 / i.sum(), 2))
geography_dfgp.rename(columns={'Count': 'Percentage'}, inplace=True)
print(geography_dfgp)
print("\n\n")

# Yüzdelik tablosunun düzenlenmiş hali
geography_dfgp = geography_dfgp.pivot_table(values = 'Percentage', index = 'Geography', columns = ['Exited'])
print(geography_dfgp)
print("\n\n")

# Coğrafyaya göre kayıp dağılımı:

labels= ['Stays', 'Exits']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

geography_dfgc.plot(kind='bar',
          color=['g', 'r'],
          rot=0, 
          ax=ax1)
ax1.legend(labels)
ax1.set_title('Churn Risk per Geography- Count', fontsize=14, pad=10)
ax1.set_ylabel('Count',size=12)
ax1.set_xlabel('Geography', size=12)


geography_dfgp.plot(kind='bar',
          color=['g', 'r'],
          rot=0, 
          ax=ax2)
ax2.legend(labels)
ax2.set_title('Churn Risk per Geography- Percentage', fontsize=14, pad=10)
ax2.set_ylabel('Percentage',size=12)
ax2.set_xlabel('Geography', size=12)

plt.show()
#Yukarıdaki istatistiklerden, en az sayıda müşterinin Almanya'dan olduğu sonucuna varabiliriz, 
#ancak görünüşe göre %32 civarında bankadan ayrılma olasılıkları en yüksek olanlardır.
print("\n\n")


print("Loss segmentation by age: ")
#Yaşa göre "Exited" bölümünü ayırıp her sınıf içindeki sıklığı görüntüleme:
age_grouped = df.groupby('Age')['Exited'].agg(Count = 'value_counts')
print(age_grouped.head())
print("\n\n")

age_dfgc = age_grouped
age_dfgc = age_dfgc.pivot_table(values = 'Count', index = 'Age', columns = ['Exited'])
print(age_dfgc.head())
print("\n\n")

#Yaşa göre "Exited" bölümünü ayırıp her sınıf içindeki sıklık yüzdesini görüntüleme:

age_dfgp = age_grouped.groupby(level=[0]).apply(lambda i: round(i * 100 / i.sum(), 2))
age_dfgp.rename(columns={'Count': 'Percentage'}, inplace=True)
print(age_dfgp.head())
print("\n\n")

# Yüzdelik tablosunun düzenlenmiş hali
age_dfgp = age_dfgp.pivot_table(values = 'Percentage', index = 'Age', columns = ['Exited'])
print(age_dfgp.head())
print("\n\n")

#Yaşa göre kayıp dağılımı:

labels= ['Stays', 'Exits']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

age_dfgc.plot(kind='bar',
          color=['g', 'r'],
          rot=0, 
          ax=ax1)
ax1.legend(labels)
ax1.set_title('Churn Risk per Age- Count', fontsize=14, pad=10)
ax1.set_ylabel('Count',size=12)
ax1.set_xlabel('Age', size=12)


age_dfgp.plot(kind='bar',
          color=['g', 'r'],
          rot=0, 
          ax=ax2)
ax2.legend(labels)
ax2.set_title('Churn Risk per Age- Percentage', fontsize=14, pad=10)
ax2.set_ylabel('Percentage',size=12)
ax2.set_xlabel('Age', size=12)

plt.show()



#df['cinsiyet'] = df['cinsiyet'].apply(str)
print("\n\n")
categories = pd.cut(df['Age'], bins=[18,25,40,65,92], labels=[ "Young","Young Adult","Middle Aged","Old"])
print(categories)
print("\n\n")

#Yaşa göre "Exited" bölümünü ayırıp her sınıf içindeki sıklığı görüntüleme:
df["AgeCategories"]=categories
age_categories_grouped = df.groupby('AgeCategories')['Exited'].agg(Count= 'value_counts')
print(age_categories_grouped)
print("\n\n")

age_categories_dfgc = age_categories_grouped
age_categories_dfgc = age_categories_dfgc.pivot_table(values = 'Count', index = 'AgeCategories', columns = ['Exited'])
print(age_categories_dfgc)
print("\n\n")

#Yaşa göre "Exited" bölümünü ayırıp her sınıf içindeki sıklık yüzdesini görüntüleme:
age_categories_dfgp = age_categories_grouped.groupby(level=[0]).apply(lambda i: round(i * 100 / i.sum(), 2))
age_categories_dfgp.rename(columns={'Count': 'Percentage'}, inplace=True)
age_categories_dfgp

#Yüzdelik tablosunun düzenlenmiş hali

age_categories_dfgp = age_categories_dfgp.pivot_table(values = 'Percentage', index = 'AgeCategories', columns = ['Exited'])
print(age_categories_dfgp.head())
print("\n\n")

#Kutu grafiği ile kategorik nitelikler dışında, çıkılan özniteliğin diğer özniteliklerle ilişkisini gösterme
print("Showing the relationship of exited attribute with other attributes except categorical attributes with boxplot:")
for i in df.select_dtypes(exclude='category').columns:
    sns.boxplot(data=df, x= 'Exited', y= i, hue='Exited')
    plt.show()
#Ürün Sayısı, Aktif Üye, Kredi kartı puanı, maaş, aboneyi terk etme olasılığı üzerinde anlamlı bir etkisi yoktur.
#Orta yaşlı müşteriler, genç olanlardan daha fazla kayıp var.
#Kullanım süresi ile ilgili olarak, her iki uçtaki müşterilerin (banka ile çok az zaman geçiren veya banka ile çok zaman geçiren)
#ortalama kullanım süresine sahip olanlara kıyasla müşteri kaybetme olasılığı daha yüksektir.
#Banka, önemli banka bakiyeleri olan müşterilerini kaybediyor.

