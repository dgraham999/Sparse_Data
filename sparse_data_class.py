import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from pyfm import pylibfm as plf
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Imputer
import pickle as pkl
seed = 73
np.random.seed(seed)

class SparseData:
	def __init__(self,data):
		self.data = data
	#implement matrix facorization
	#create factor program  dictionaries
	def melt_dict(self):
		#define new lists
		X = []
		y = []
		Xpred = []
		#remove ID parameter and add T paramter as time periods
		self.data.drop(['ID'],axis=1)
		t = list(range(0,self.data.shape[0]))
		self.data['T'] = t
		#melt into single dictionary vector on T as key
		data_melt = pd.melt(self.data,id_vars=['T'])
		#identify training set using notnull on melt vector
		not_null_vec = pd.notnull(data_melt.value)
		#identify prediction set using isnull on melt vector
		null_vec = pd.isnull(data_melt.value)
		#create training and prediction vectors
		data_train = data_melt[not_null_vec]
		data_predict = data_melt[null_vec]
		#convert data vectors into lists
		X = []
		y = []
		Xpred = []
		#create dictionaries of each column and rank entered
		for i in range(len(data_train)):
			X.append({'response_id':str(data_train.iloc[i,0]),'ques_id':str(data_train.iloc[i,1])})
			y.append(float(data_train.iloc[i,2]))

		for i in range(len(data_predict)):
			Xpred.append({'response_id': str(data_predict.iloc[i,0]),'ques_id': str(data_predict.iloc[i,1])})
		return X, y, Xpred, data_train, data_predict

	def apply_fm(self, X, y, Xpred):
		#create test splits for mse calculation
		Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,random_state=seed, train_size = .9)
		#vectorize the dictionary
		v= DictVectorizer()
		X_train = v.fit_transform(Xtrain)
		X_test = v.transform(Xtest)
		X_pred = v.transform(Xpred)
		#create instance of plf
		fm = plf.FM(num_factors=20, num_iter=30, verbose=True, task="regression", initial_learning_rate=0.01, learning_rate_schedule="optimal")
		#run training data through matrix factorization
		fm.fit(X_train,ytrain)
		#make predictions on test set
		y_test = fm.predict(X_test)
		#compute mse on test set
		mse = mean_squared_error(ytest, y_test)
		#compute predicted values for nan set
		y_pred = fm.predict(X_pred)
		return y_pred, mse

	def merge_data(self, data_predict, y_pred, data_train):
		#replace values in vector with predicted values
		data_predict = data_predict.drop(['value'],axis=1)
		data_predict.loc[:,'value']=y_pred
		#concatenate vectors vertically and sort by index
		dfx = pd.concat([data_train,data_predict])
		#dfx = dfx.sort_index()
		#pivot vector back into data table
		dfx = dfx.pivot_table(index = ['response_id'], columns=['variable'])
		#remove multi index
		dfx = pd.DataFrame(dfx.values, index=dfx.index, columns=dfx.columns.levels[1])
		#remove columns names 'variable'
		dfx.columns.names = [None]
		dfx.reset_index(drop=True)
		
		return dfx
