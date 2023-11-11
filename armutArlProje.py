
#################################

# İş Problemi
#################################


# Türkiye’nin en büyük online hizmet platformu olan Armut, hizmet verenler ile hizmet almak isteyenleri buluşturmaktadır.
# Bilgisayarın veya akıllı telefonunun üzerinden birkaç dokunuşla temizlik, tadilat, nakliyat gibi hizmetlere kolayca
# ulaşılmasını sağlamaktadır.
# Hizmet alan kullanıcıları ve bu kullanıcıların almış oldukları servis ve kategorileri içeren veri setini kullanarak
# Association Rule Learning ile ürün tavsiye sistemi oluşturulmak istenmektedir.

#################################
# IMPORT AND DATA PREPROCCESSİNG
#################################

import pandas as pd
import datetime as dt
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', None)

df_ = pd.read_csv('datasets/2-recommendationSystems/armut_data.csv')
df = df_.copy()
df.head()
df.info()
df.isnull().sum()
df.UserId.nunique() # 24826 benzersiz user_id var bu veri setinde yani bazı müşteriler doğal olarak birden fazla hizmet almış
df.groupby('UserId').agg({'ServiceId': 'count'}) # burada user_idlerin 0'dan 24826 kadar gittigini anlıyoruz ve user_idleri groupby ile gruplandırarak kişilerin kaçar servis aldığını görüyoruz
df.groupby('UserId').agg({'ServiceId': 'count'}).sort_values(by= 'ServiceId', ascending=False).head()
df.groupby('UserId').agg({'ServiceId': 'count'}).sort_values(by= 'ServiceId', ascending=False).tail()

###################################
# ServisID her bir CategoryID özelinde farklı bir hizmeti temsil etmektedir. ServiceID ve CategoryID’yi "_" ile birleştirerek bu hizmetleri
# temsil edecek yeni bir değişken oluşturunuz
###################################
dff = df.copy() # kullanmak için kopyaladım
df['hizmet'] = df['ServiceId'].astype(str) + '_' + df['CategoryId'].astype(str)


###################################
# Bunun için öncelikle sadece yıl ve ay içeren yeni bir
# date değişkeni oluşturunuz. UserID ve yeni oluşturduğunuz date değişkenini "_" ile birleştirirek ID adında yeni bir değişkene atayınız.
###################################
def convert_date_year_and_month(df):
    df['CreateDate'] = pd.to_datetime(df['CreateDate'])
    df['sepetID'] = pd.Series([date.strftime('%Y-%m') for date in df.CreateDate])
    df['sepetID'] = df['UserId'].astype(str) + '_' + df['sepetID'].astype(str)
    df['new_date'] = pd.Series([date.strftime('%Y-%m') for date in df.CreateDate])
    df.head()
    return df
df = convert_date_year_and_month(df)


####################################
# CREATE PIVOT-TABLE
####################################
df_pro = df.groupby(['sepetID', 'hizmet'])['hizmet'].count(). \
    unstack(). \
    fillna(0). \
    applymap(lambda x: 1 if x > 0.0 else 0)

####################################
# Association Rule
####################################
def creat_rules(df_pro):
    frequency_items = apriori(df_pro,
                              min_support=0.01,
                              use_colnames=True
                              )
    frequency_items.sort_values(by='support',  ascending=False)
    rules = association_rules(frequency_items,
                              metric='support',
                              min_threshold=0.01)

    rules = rules.sort_values(by='lift', ascending=False)
    return rules

rules = creat_rules(df_pro)
#####################################
# Arl_reccomender
#####################################

def arl_reccomender(rules_df, product_id, rect=1):
    sorted_rules = rules_df.sort_values(by='lift', ascending=False)
    reccomend_list = []
    for i, hizmet in enumerate(sorted_rules['antecedents']):
        for j in list(hizmet):
            if j == product_id:
                reccomend_list.append(list(sorted_rules.iloc[i,]['consequents']))

    reccommend_list = list({item for item_list in reccomend_list for item in item_list})
    return reccommend_list[:rect]

arl_reccomender(rules, product_id= '2_0', rect=3)

######################################
# Tavsiye (Recommendation)
######################################
'''
        Türkçe
sevgili armut kullanıcısı umarım aldıgınız '2_0' hizmetinden memnun kalmışsınızdır. Diğer armut kullanıcılardan gözlemlediğimiz, 
analiz ettiğimiz kadarıyla '2_0' hizmetini alan kullanıcıların ['25_0', '13_11', '15_1'] hizmetlerini aldıkları da gözlenmiştir.
Siz değerli kullancımızında böyke bir hizmet ihtiyacı olabilir diye sizlere ['25_0', '13_11', '15_1'] hizmetleri %10 indirimle sunuyoruz 
sevgilerimizle...
------------------------------------------------------------------------------------------------------------------------
        Engilish 
Hello, dear Armut user, I hope you have been satisfied with the '2_0' service you received. Based on our observations and analysis of other 
Armut users, it has been noticed that users who have used the '2_0' service have also availed themselves of the 
['25_0', '13_11', '15_1'] services. Considering that you, our valued user, may also have a need for such services, we are offering you the 
['25_0', '13_11', '15_1'] services with a 10% discount. With warm regards...

'''




























































































































































