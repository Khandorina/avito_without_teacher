import pandas as pd
import re
import numpy as np

#################ОЧИСТКА ДАННЫХ И ПРЕОБРАЗОВАНИЕ для классификатора###############################
df = pd.read_excel(io='goodparser.xlsx', engine='openpyxl')

df = df.drop(['Телефон','Город', 'Ссылка', 'Фейк'], axis=1)

pd.set_option('display.max_columns', None)

#функция чтобы найти конкретно пользователей. Они чаще всего под именем и фамилией или только имя(1 или 2 слова). Делаем такое допущение, иначе хз
def names(s):
    agencies = ["Агентство недвижимости «СОВА»","\"Агенство Недвижимости и компания\"","АН  \"Комфорт Сити\"","Этажи Тюмень","Variantmn","Фаворит +","Небоскреб tmn","Мегаполис","АН Квадратный метр"]
    if s not in agencies:
        if len(s.split())==1 or len(s.split())==2 :
            s = "пользователь"
    return s

df["Имя контакта"] = df["Имя контакта"].apply(lambda x: names(str(x).lower()))

df = pd.get_dummies(df, columns=['Имя контакта'])
df = pd.get_dummies(df, columns=['Название'])


def clean_streets(s):
    #удаляю цифры в адресе. думаю, это не так важно. главное - само название
    s = re.sub(r'[^\w\s]+|[\d]+', r'',s).strip()
    #удаляю "ул" и другие мелкие слова
    shortword = re.compile(r'\W*\b\w{1,3}\b')
    s = shortword.sub('', s)
    s = ' '.join(word for word in s.split() if word not in ["Тюменская", "область", "Тюменский","Патрушева", "Микрорайон"])
    return s
df["Адрес"] = df["Адрес"].apply(lambda x: clean_streets(str(x)))
df = pd.get_dummies(df, columns=['Адрес'])

#создаю колонки просто, потом их заполняю
df[["Этаж","Комиссия","Количество комнат", "Размер комссии", "Залог"]] = np.random.randint(10, size=(len(df), 5))

def clean_additional(s):
    s = s.split(';')
    lst1 = []

    for _ in s:
        ss = ' '.join(word for word in _.split() if word not in ["<br>"])
        lst1.append(ss)
    lst2=[]
    for l in lst1:
        lst2.append(l.split("=>"))
    return lst2

for i in range(0, df.shape[0]):
    clean_lst = clean_additional(str(df.iloc[i]['Дополнительная информация']))

    for _ in clean_lst:
        if _[0] == 'Комиссия':
            if _[1] == 'Есть':
                _[1] = 1
            else:
                _[1] = 0
        if _[0] == 'Количество комнат':
            if _[1] == 'Студия':
                _[1] = 0

    for _ in clean_lst:
        if _[0] in df.columns:
            df.at[i, _[0]] = _[1]

df = df.drop(['Дополнительная информация'], axis=1)


#print(df)

#################КЛАСТЕРИЗАЦИЯ###############################
##################данных много, использую PCA для уменьшения размерности. то же самое и для деревьев можно было сделать кстати#####################################

X = df

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X = sc.fit_transform(X)

from sklearn.decomposition import PCA

#21 первые компоненты имеют самые большие значения, в дальнейшем использую это число
#pca = PCA()
#X = pca.fit_transform(X)
#explained_variance = pca.explained_variance_ratio_
#print(explained_variance)

pca = PCA(n_components=10)
X = pca.fit_transform(X)
#explained_variance = pca.explained_variance_ratio_
#print(explained_variance)


import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

kmeans_kwargs = {
    "init": "k-means++",
    "random_state": 1,
}

#создать список для хранения значений SSE для каждого k
sse = []
for k in range (1, 21):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(X)
    sse.append(kmeans.inertia_)

#визуализируйте результаты, например,график(диапазон (1, 11), sse)
plt.plot(range(1, 21), sse)
plt.xticks(range(1, 21))
plt.xlabel("Количество кластеров")
plt.ylabel("SSE")
plt.show()