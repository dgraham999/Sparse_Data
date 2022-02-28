
import pandas as pd
from pandas import DataFrame
from pandas import concat
from sparse_data_class import SparseData

def shift_shape(data, lag=1, targets=1):
	"""
	Frame a time series as a supervised learning dataset.
	Arguments:
		data: Sequence of observations as a list or dataframe.
		lag: Number of lag observations as input (X).
		targets: Number of observations as output (y).
		dropnan: Boolean whether or not to drop rows with NaN values.
	Returns:
		Pandas DataFrame framed for time lags and targets.
	"""

	df = data.drop(drops,axis=1,inplace=False)
	n_vars = 1 if type(df) is list else df.shape[1]
	cols, names = list(), list()
	col_names = df.columns
	# input sequence (t-n, ... t-1)
	for i in range(lag, 0, -1):
		cols.append(df.shift(i))
		names += [col_names[j]+'_lag(t-'+str(i)+')' for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, targets):
		cols.append(df.shift(-i))
		if i == 0:
			names += [col_names[j]+'_target(t)' for j in range(n_vars)]
		else:
			names += [col_names[j]+'_target(t+'+str(i)+')' for j in range(n_vars)]
	# put it all together
	data_shift = concat(cols, axis=1)
	data_shift.columns = names
	#call sparse data class
	sd = SparseData(data_shift)
	X, y, Xpred, data_train, data_predict, data_reg, data_resp = sd.melt_dict()
	y_pred, mfmse = sd.apply_fm(X, y, Xpred)
	data_new = sd.merge_data(data_predict, y_pred, data_train, data_reg)
	return data_new
