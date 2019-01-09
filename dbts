# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 18:37:35 2018

@author: Beyza
"""


import pandas as pd    # verileri düzgün bir şekilde tutabilmek ve dataframe oluşturabilmek için kullanılan kütüphane
from sklearn.preprocessing import LabelEncoder, OneHotEncoder   # Her bir değer için bir sayısal değer veren Encoder = LabelEncoder
# Encoderleri sütun haline getiren = OneHotEncoder
from sklearn.cross_validation import train_test_split # verilerin eğitim ve test kısmı olarak ayrılması için kullanılan kütüphane
from sklearn.preprocessing import StandardScaler # Verileri Scaler haline getirme işleminin yapılması için kullanılan kütüphane
from sklearn.neighbors import KNeighborsClassifier   # KNN sınıflandırıcısını kullanarak en yakın komşu ile verileri tahmin ettirme işlemi


# Veri Yukleme
veriler = pd.read_csv('diziler.csv')
# =============================================================================
# veriler["Bitis Yili"].fillna("2018", inplace = True)    # NaN değerlerin içerisini doldurma işlemi
# veriler["Sezon Sayisi"].fillna("1", inplace = True)
# veriler["Yayinlanan bolum sayisi"].fillna("15", inplace = True)
# veriler["ihrac durumu"].fillna("0", inplace = True)
# =============================================================================


# veri ön işleme
# Encoder işleminin yapılması için numeric olmayan değerlerin kolonlara ayrılması işlemi
le = LabelEncoder()    # LabelEncoder objesi
diziAdi = veriler.iloc[:,0:1].values
baslangicYilOrtSure = veriler.iloc[:,1:3].values
oyuncular = veriler.iloc[:,8:11].values

yayinciKurulus = veriler.iloc[:,3:4].values
yayinciKurulus[:,0] = le.fit_transform(yayinciKurulus[:,0])  # fit_transform hem uygulasın hem de sonucun ulkenin içine yazılması işlemi

yapimSirketi = veriler.iloc[:,4:5].values
yapimSirketi[:,0] = le.fit_transform(yapimSirketi[:,0])     

yapimci = veriler.iloc[:,5:6].values
yapimci[:,0] = le.fit_transform(yapimci[:,0])     

senarist = veriler.iloc[:,6:7].values
senarist[:,0] = le.fit_transform(senarist[:,0])     

yonetmen = veriler.iloc[:,7:8].values
yonetmen[:,0] = le.fit_transform(yonetmen[:,0])     
 
turler = veriler.iloc[:,11:12].values
turler[:,0] = le.fit_transform(turler[:,0])     

bitisYili = veriler.iloc[:,12:13].values
sezonSayisi = veriler.iloc[:,13:14].values
bolumSayisi = veriler.iloc[:,14:15].values
ihracDurumu = veriler.iloc[:,15:].values


ohe = OneHotEncoder(categorical_features="all")
 # hangi categorical features'ları alacağımızı belirleriz
 
yayinciKurulus = ohe.fit_transform(yayinciKurulus).toarray()   # daha önceden veri kümesinde ayırdığımız verileri OneHot ile 0 ve 1 olmak üzere
# tek tek sütunlar haline getiririz
yapimSirketi = ohe.fit_transform(yapimSirketi).toarray()
yapimci = ohe.fit_transform(yapimci).toarray()
senarist = ohe.fit_transform(senarist).toarray()
yonetmen = ohe.fit_transform(yonetmen).toarray()
turler = ohe.fit_transform(turler).toarray()


 # verileri DataFrame haline getirme ve sütun isimlerinin verilmesi
diziAdiSonuc = pd.DataFrame(data = diziAdi, index = range(184), columns = ['Dizi Adı']) 
baslangicYilOrtSureSonuc =pd.DataFrame (data = baslangicYilOrtSure, index = range(184), columns =['Başlangıç Yılı','Ortalama Süresi']) 
yayinciKurulusSonuc = pd.DataFrame(data = yayinciKurulus, index = range(184), columns=['ATV','Fox Tv','Kanal D', 'Show Tv', 'Star Tv','TRT 1'])
yapimSirketiSonuc = pd.DataFrame(data = yapimSirketi, index = range(184), columns = ['25 Film','ANS Yapim','Adam Film','Altioklar Film','Arzu Film',
                                  'Asis Film','Avsar Film',' Ay Yapim',' BKM','BSK Yapim','Bando Yapim','Barakuda Yapim','Bi Yapim','Boyut Film',
                                  'D Productions','ES Film','Eflatun Film','Endemol Shine','Erler Film','Focus Film',' Gold Film','Koliba Film',
                                  'Kuzey Productions','Limon Film','LuckyRed','MF Yapim','Med Yapim','Mia Yapim','Mint Film','Most Production',
                                  'NTC Medya','No Dokuz Productions','03 Medya','Origami Yapim','Ortaks Yapim','Pana Film',
                                  'Pastel Film','Pusula Film','Salacak Yapim','Scor Film','Set Film','Sinegraf','Sinetel','Sis Yapim',
                                  'Stil Medya','Surec Film','TMC','Tekden Film','Tims Productions','TukenmezKalem Film','Us Yapim'])
yapimciSonuc = pd.DataFrame(data = yapimci, index = range(184))
senaristSonuc = pd.DataFrame(data = senarist, index = range(184))
oyuncularSonuc = pd.DataFrame(data = oyuncular, index = range(184))
turlerSonuc = pd.DataFrame(data = turler, index = range(184), columns =['Aksiyon','Dram','Fanstastik','Genclik','Komedi',
                            'Polisiye','Romantik','Tarihi'])
bitisYiliSonuc = pd.DataFrame(data = bitisYili, index= range(184), columns= ['Bitiş Yılı'])
sezonSayisiSonuc = pd.DataFrame(data = sezonSayisi, index= range(184), columns = ['Sezon Sayısı'])
bolumSayisiSonuc =pd.DataFrame(data = bolumSayisi, index= range(184), columns=['Bölüm Sayısı'] )
ihracDurumuSonuc = pd.DataFrame(data=ihracDurumu[:,:1], index = range(184), columns=['İhraç Durumu'])


# verilerin birleştirilmesi
s1 = pd.concat([baslangicYilOrtSureSonuc,yayinciKurulusSonuc,yapimSirketiSonuc,yapimciSonuc,
                senaristSonuc,oyuncularSonuc,turlerSonuc], axis =1)
s2 = pd.concat([bitisYiliSonuc, sezonSayisiSonuc, bolumSayisiSonuc, ihracDurumuSonuc], axis = 1)

# verilerin eğitim ve test kısmı olarak ayrılması
x_train, x_test, y_train, y_test = train_test_split(s1,s2, test_size=0.33)  # random_state = nasıl böleceğini belirleriz
# veriler test_size = 0.33 ile veri kümesi 1/3 test 2/3 eğitim olacak şekilde iki bölüme ayrılır

# Verileri Scaler haline getirme işleminin yapılması
sc = StandardScaler()
X_train = sc.fit_transform(x_train)      # fit eğitme, transform ise o eğitimi kullanma ve uygulama işlemini gerçekleştirir
X_test = sc.fit_transform(x_test)
Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)
 
# Verileri eğiterek tahmin işlemini gerçekleştirme 
knn = KNeighborsClassifier(n_neighbors=2, metric='euclidean')  # yakın komşu sayısını ve metric ile komşu uzaklığının
#nasıl hesaplanacağının yöntemini belirleriz
# en yakın komşu sayısını 2 ve mesafe ölçümü yöntemini öklid cinsinden verdiğimizde en optimal sonuca ulaşmış oluruz
knn.fit(X_train,y_train)  # KNN ile X_train'den y_train'i öğrenme işlemi
y_pred = knn.predict(X_test)   # X_test kümesini tahmin ettirme işlemi


 
