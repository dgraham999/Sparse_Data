# Sparse_Data
There are currently 2 files in this repository.  One is the class sparse_data_class.py and the other is shift_shape.py.  The sparse data class is intended to impute missing data using the pylibfm, a bayes optimizer, library and is to be used when a parallel computing source is not available.  The neural network version of this will be posted shortly.  The shift_shape.py class is a simple utility to shift data for lags and leads but it demonstrates the use of the sparse data class since shifts always create empty cells during the shift.  The commentary in the code should document it sufficiently. 
