# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from konlpy.tag import Okt
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from sklearn.preprocessing import LabelEncoder
# from tensorflow.keras.utils import to_categorical
# import pickle
# from tensorflow.keras.models import load_model
#
# df = pd.read_csv('./crawling_data/naver_headline_news_20240125.csv')
# print(df.head())
# df.info()
#
# X=df['titles']
# Y=df['category']
#
#
# with open('./models/label_encoder.pickle','rb') as f:
#     label_encoder = pickle.load(f)
# labeled_y = label_encoder.transform(Y)
# label = label_encoder.classes_
# print(labeled_y)
# print(label)
#
# okt = Okt()
#
# for i in range(len(X)):
#     X[i]=okt.morphs(X[i],stem=True)
#
# stopwords = pd.read_csv('./stopwords.csv', index_col=0)
# for j in range(len(X)):
#     words = [] # j는 문장
#     for i in range(len(X[j])):
#         if len(X[j][i]) > 1:
#             if X[j][i] not in list(stopwords['stopword']):
#                 words.append(X[j][i])
#     X[j] = ' '.join(words)
# #print(X[:5]) #불필요한 문자들을 삭제하고 리스트에 저장
# with open('./models/news_token.pickle','rb') as f:
#     token = pickle.load(f)
# #각 단어에 번로를 부여 (같은 형태소에 같은 번호를 붙여 주었다.)
# token.fit_on_texts(X) #라벨을 붙이기
# tokened_x = token.texts_to_sequences(X) #문장을 숫자 번호로 만듬
#
# for i in range(len(tokened_x)):
#     if len(tokened_x[i]) > 27:
#         tokened_x[i]= tokened_x[i][:27]
# print(tokened_x)
#
# x_pad = pad_sequences(tokened_x,27)
#
# model = load_model('C:/pythoncharm/news_category_classification_intel_team2/models/news_catrgory_calssification_model_0.8349100947380066.h5')
# preds = model.predict(x_pad)
# # wordsize = len(token.word_index) + 1 #index가 1붙어 만들어 진다. +1를 더한 이유는 0을 쓰기 위함
# # #print(tokened_x)
# # print(wordsize) #사실 모든 총 숫자는 33개
#
# predicts = []
# for pred in preds:
#     most = label[np.argmax(pred)]
#     pred[np.argmax(pred)]=0
#     second = label[np.argmax(pred)]
#     predicts.append([most,second])
# df['predict']=predicts
#
# print(df)
# exit()
# df['OX'] = 0
# for i in range(len(df)):
#     if df.loc[i,'category'] in df.loc[i,'predict']:
#         df.loc[i,'OX'] = 'O'
#     else:df.loc[i,'OX']='X'
# print(df['OX'].value_counts())
# print(df['OX'].value_counts()/len(df))
#
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import pickle
from tensorflow.keras.models import load_model

df = pd.read_csv('./crawling_data/naver_headline_news_20240125.csv')
print(df.head())
df.info()

X = df['titles']
Y = df['category']

with open('./models/label_encoder.pickle','rb') as f:
    label_encoder = pickle.load(f)

label = label_encoder.classes_

print(label)

okt = Okt()

for i in range(len(X)):
    X[i] = okt.morphs(X[i], stem=True)

stopwords = pd.read_csv('./stopwords.csv', index_col=0)
for j in range(len(X)):
    words = []
    for i in range(len(X[j])):
        if len(X[j][i]) > 1:
            if X[j][i] not in list(stopwords['stopword']):
                words.append(X[j][i])
    X[j] = ' '.join(words)

with open('./models/news_token.pickle', 'rb') as f:
    token = pickle.load(f)
tokened_x = token.texts_to_sequences(X)
for i in range(len(tokened_x)):
    if len(tokened_x[i]) > 27:
        tokened_x[i] = tokened_x[i][:27]
print(tokened_x)

x_pad = pad_sequences(tokened_x, 27)

model = load_model('./models/news_catrgory_calssification_model_0.8349100947380066.h5')
preds = model.predict(x_pad)

predicts = []
for pred in preds:
    most = label[np.argmax(pred)]
    pred[np.argmax(pred)] = 0
    second = label[np.argmax(pred)]
    predicts.append([most, second])
df['predict'] = predicts

print(df)

df['OX'] = 0
for i in range(len(df)):
    if df.loc[i, 'category'] in df.loc[i, 'predict']:
        df.loc[i, 'OX'] = 'O'
    else:
        df.loc[i, 'OX'] = 'X'
print(df['OX'].value_counts())
print(df['OX'].value_counts()/len(df))

















