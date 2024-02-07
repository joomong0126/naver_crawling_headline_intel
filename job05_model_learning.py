# import numpy as np
# import matplotlib.pyplot as plt
# from tensorflow.keras.models import *
# from tensorflow.keras.layers import *
#
# X_train, X_test, Y_train, Y_test = np.load('./news_data_max_3_wordsize_8642.npy',allow_pickle=True)
# print(X_train.shape, Y_train.shape)
# print(X_test.shape, Y_test.shape)
#
# model = Sequential()
# model.add(Embedding(1191, 300, input_length=3)) # 1191를 300차원으로 줄이겠다는 의미.
# model.add(Conv1D(32,kernel_size=5,padding='same',activation='relu'))
# model.add(MaxPooling1D(pool_size=1)) # 1로 두는 것이 좋음.
# model.add(LSTM(128,activation='tanh',return_sequences=True)) # return_sequences
# # LSTM 셀이 있어. LSTM에
# model.add(Dropout(0.3))
# model.add(LSTM(64,activation='tanh',return_sequences=True))
# model.add(Dropout(0.3))
# model.add(LSTM(64,activation='tanh')) # 마지막 LSTM에만 return Sequence를 안 줌.
# model.add(Dropout(0.3))
# model.add(Flatten())
# model.add(Dense(128,activation='relu'))
# model.add(Dense(6,activation='softmax')) # 돌려봤을 때 가장 좋았던 조건으로 교수님께서 설정해서 알려주심.
# model.summary()
#
# model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
# fit_hist = model.fit(X_train,Y_train,batch_size=128,epochs=10,validation_data=(X_test,Y_test))
# model.save('./models/news_category_classification_model_{}.h5'.format(fit_hist.history['val_accuracy'][-1]))
# plt.plot(fit_hist.history['val_accuracy'], label='validation accuracy')
# plt.plot(fit_hist.history['accuracy'], label='train accuracy')
# plt.legend()
# plt.show()
# # 전체 문장을 학습을 해서 좌표화함. 그래서 비슷한 의미를 가진 애들을 따로 묶어줌.
# # 백터화를 해서 2차원 공간안에서 단어들이 어떻게 있는지를 알 수 있는 학습을 하겠다. -> 시각화
# # 20명이 우주 공간에 있다고 하면 -> 있는 듯 없는 듯 해짐 -> 밀도가 낮아짐 -> 이것을 희소해진다고 해. -> 데이터가 희소해진다. -> 차원축소-> 데이터의 성질을 최대한 유지함.
# # 데이터들 간의 관계를 하나하나 축소 시킴.
import  numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import *
from  tensorflow.keras.layers import *

X_train, X_test, Y_train, Y_test = np.load(
    './news_data_max_27_wordsize_11228.npy', allow_pickle=True)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

model = Sequential()
model.add(Embedding(11228, 300, input_length=27))#자연어 의미를 학습하는 레이어
model.add(Conv1D(32, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=1))
model.add(LSTM(128, activation='tanh', return_sequences=True)) #정가 하나만 있을때는 사용안함 return
model.add(Dropout(0.3))
model.add(LSTM(64, activation='tanh', return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(64, activation='tanh'))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(6, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
fit_hist = model.fit(X_train, Y_train, batch_size=128, epochs= 10, validation_data=(X_test, Y_test))
model.save('./models/news_catrgory_calssification_model_{}.h5'.format(fit_hist.history['val_accuracy'][-1]))
plt.plot(fit_hist.history['val_accuracy'], label='validation accuracy')
plt.plot(fit_hist.history['accuracy'], label='train accuracy')
plt.legend()
plt.show()