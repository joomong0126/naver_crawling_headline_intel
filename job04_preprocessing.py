# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from konlpy.tag import Okt  #자연어 처리하는 패키지 한국어라서 ko임
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from sklearn.preprocessing import LabelEncoder
# from tensorflow.keras.utils import to_categorical
# import pickle
#
#
# df = pd.read_csv('./naver_news_titles_20240125.csv')
# print(df.head())
# df.info()
#
# X = df['titles']
# Y = df['category']
#
# label_encoder= LabelEncoder()
# labeled_y = label_encoder.fit_transform(Y)
# print(labeled_y[:3])
# label = label_encoder.classes_ # classes_를 보면 돼. ( label_encoder을 pickle로 담궈 놓음)
# print(label)
# with open('./models/label_encoder.pickle','wb') as f:
#     pickle.dump(label_encoder, f)   # 이걸실행하면 models 폴더에 label_encoder.pickle이라는 폴더가 열음, wb(writebinary)
#     # pickle : 원래 그 형태 그대로 가져옴 ( 피클을 담그면 나중에 먹어도 그 맛 그대로 가져옴 )
# onehot_y = to_categorical(labeled_y)
# print(onehot_y[:3])
# print(X[1:5])
# okt = Okt()
# temp = []
# for i in range(len(X)):
#     X[i] = okt.morphs(X[i], stem=True)  #stem 을 줘야 원형으로 바뀜
#     if i % 1000:
#         print(i)
# # 0       한동훈  민주당 운동권에 죄송한 마음 전혀 없어 청년에겐 죄송함 커   Politics
# # 어절 하나하나에다가 토큰을 붙여주면 너무 말이 많으니까, 쪼개진 애들이 형태소 야.
# # 그 작업을 하기 위해서 필요한 게 Okt 야. okt: 형태소 분리를 해주는 거야.
# #감탄사는 의미 학습에서 의미가 없고 오히려 방해만 되서 제거해 줘야함. 이렇게 모델이 학습하는데 필요없는 단어를 불용어라고함.
#
# stopwords = pd.read_csv('./stopwords.csv', index_col=0)
# for j in range(len(X)):      #한글자 짜리 제거하고 불용어 제거하고
#     words = []
#     for i in range(len(X[j])):
#         if len(X[j][i]) > 1:
#     # 형태소 단위로 접근하기 위해 2개의 인덱스가 필요함.
#             if X[j][i] not in list(stopwords['stopword']):
#                 words.append(X[j][i])
#     X[j] = ''.join(words)
# print(X)    #최종 완성된 X임
#
# token = Tokenizer()
# token.fit_on_texts(X) # x 안에 있는 모든 토큰을 보내줘..?
# tokened_x = token.texts_to_sequences(X)
# wordsize = len(token.word_index)+1
# print(tokened_x)
# print(wordsize)
#
# with open('./models/label_encoder.pickle','wb') as f:
#     pickle.dump(token, f)
#
# # lstm을 쓰지만, 의미있는 단어들은 뒤쪽에 배치하고 아무것도 없는 것은 앞쪽에 0을 입력시킬거에요.
# max = 0
# for i  in range(len(tokened_x)):
#     if max < len(tokened_x[i]):
#         max = len(tokened_x[i])
# print(max)
#
# x_pad = pad_sequences(tokened_x,max)
# print(x_pad)
#
# X_train,X_test,Y_train,Y_test = train_test_split( x_pad, onehot_y, test_size = 0.2)
# print(X_train.shape, Y_train.shape)
# print(X_test.shape, Y_test.shape)
#
# xy = X_train, X_test,Y_train,Y_test
# xy=np.array(xy,dtype=object)
# np.save('./news_data_max_{}_wordsize_{}'.format(max, wordsize),xy)
#
#
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

df = pd.read_csv('./naver_news_titles_20240125.csv')
print(df.head())
df.info()

X = df['titles']
Y = df['category']

label_encoder = LabelEncoder()
labeled_y = label_encoder.fit_transform(Y) #라벨 부여
print(labeled_y[:3])
label = label_encoder.classes_  #부여된 라벨 정보를 확인
print(label)
with open('./models/label_encoder.pickle', 'wb') as f:
    pickle.dump(label_encoder, f) #파이썬 데이터형으로 저장 리스트 -> 리스트로 불러온다.
onehot_y = to_categorical(labeled_y)
print(onehot_y[:3])
print(X[:5])
okt = Okt()
#for i in range(len(X[:5])):
 #   X[i] = okt.morphs(X[i])  #그냥 짜르기만 한다.
for i in range(len(X)):
    X[i] = okt.morphs(X[i], stem=True) #원형으로 바꾸어 주고
    if i % 1000:
        print(i)
#print(X[:5]) #한글자는 학습이 불가능, 접속사, 감탄사 는 의미 없음


stopwords = pd.read_csv('./stopwords.csv', index_col=0)
for j in range(len(X)):
    words = [] # j는 문장
    for i in range(len(X[j])):
        if len(X[j][i]) > 1:
            if X[j][i] not in list(stopwords['stopword']):
                words.append(X[j][i])
    X[j] = ' '.join(words)
#print(X[:5]) #불필요한 문자들을 삭제하고 리스트에 저장

#각 단어에 번로를 부여 (같은 형태소에 같은 번호를 붙여 주었다.)
token = Tokenizer()
token.fit_on_texts(X) #라벨을 붙이기
tokened_x = token.texts_to_sequences(X) #문장을 숫자 번호로 만듬
wordsize = len(token.word_index) + 1 #index가 1붙어 만들어 진다. +1를 더한 이유는 0을 쓰기 위함
#print(tokened_x)
print(wordsize) #사실 모든 총 숫자는 33개

with open('./models/news_token.pickle', 'wb') as f:
    pickle.dump(token, f)

max = 0 #최대값을 찾는 코드
for i in range(len(tokened_x)):
    if max < len(tokened_x[i]):
        max = len(tokened_x[i])
print(max)

x_pad = pad_sequences(tokened_x, max) #앞에 다가 0을 채워 길이를 맞추기 위함 (뒤로 가는게 학습에 좋기 때문에)
print(x_pad)

X_train, X_test, Y_train, Y_test = train_test_split(
    x_pad, onehot_y, test_size = 0.2)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

xy = X_train, X_test, Y_train, Y_test
xy = np.array(xy, dtype=object)
np.save('./news_data_max_{}_wordsize_{}'.format(max, wordsize), xy) #저장