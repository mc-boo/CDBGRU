import numpy
import math
from pandas import read_csv
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, Convolution1D, Activation, Bidirectional, GRU, RNN, MaxPooling1D, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import keras
import tensorflow as tf
from tensorflow.keras import backend as K
import csv


def create_dataset(dataset, look_back=7):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0:7]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0:7])
	return numpy.array(dataX), numpy.array(dataY)

	for i in range(1,tf.shape(y_true)):
		delta_y_true[i] = y_true[i + 1] - y_true[i]
		delta_y_pred[i] = y_pred[i + 1] - y_pred[i]
	loss1 = K.mean(K.square(y_pred - y_true), axis=-1)
	loss2 = K.mean(K.square(delta_y_pred - delta_y_true), axis=-1)
	loss3 = K.mean(K.abs(y_pred - y_true), axis=-1)
	return loss1 + loss2 + loss3



dataframe = read_csv('tmp.csv', usecols=[0, 1, 2, 3, 4, 5, 6], engine='python', skipfooter=3)
dataset = dataframe.values
dataset = dataset.astype('float32')
dataset = dataset[:, 0:7]
dataset1 = dataset


scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# split into train and test sets
look_back = 7
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size + look_back
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
# reshape into X=t and Y=t+1

trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 7))
testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 7))

model = Sequential()
model.add(Convolution1D(filters=20, kernel_size=3, padding='Same', activation='relu',  input_shape=(look_back, 7)))
model.add(MaxPooling1D(pool_size=2, strides=2))
model.add(Convolution1D(filters=10, kernel_size=3, padding='Same', activation='relu',  input_shape=(look_back, 7)))
model.add(MaxPooling1D(pool_size=2, strides=2))
model.add(Bidirectional(GRU(80, input_shape=(look_back, 7))))
model.add(Dropout(0.2))
#model.add(LSTM(40, input_shape=(look_back, 7)))
model.add(Dense(7))
#model.add(Dropout(0.25))
#model.add(Dense(7))
model.compile(loss= 'mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=50, verbose=2)
# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
print(numpy.shape(trainPredict))
print('Test R2_score: %.4f r2' % (r2_score(testY, testPredict)))
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform(trainY)
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform(testY)
# calculate root mean squared error
print('Test R2_score: %.4f r2' % (r2_score(testY, testPredict)))
trainScore = math.sqrt(mean_squared_error(trainY[:, 6], trainPredict[:, 6]))
print('Train Score: %.2f RMSE' % (trainScore))
trainScore = math.sqrt(mean_absolute_error(trainY[:, 6], trainPredict[:, 6]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(r2_score(testY[:, 6], testPredict[:, 6]))
print('Test Score: %.2f RMSE' % (testScore))
k = 0
testYlabel = testY[:, 6]
piancha = testY[:, 6]- testPredict[:, 6];
for i in range (1, len(testYlabel)):
	if testYlabel[i] != 0:
		if math.fabs(piancha[i]/testYlabel[i]) > 0.3:
			k = k+1
print(k/len(testYlabel))


numpy.savetxt('./trainPredict.txt', trainPredict)
numpy.savetxt('./trainY.txt', trainY)
numpy.savetxt('./testPredict.txt', testPredict)
numpy.savetxt('./testY.txt', testY)

trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, 0] = trainPredict[:, 6]
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, 0] = testPredict[:, 6]
# plot baseline and predictions

plt.plot(dataset1[500:1000, 0])
plt.plot(trainPredictPlot[500:1000, 0]+24)
#plt.plot(testPredictPlot[:, 0])
plt.show()