'''    2014 Phong Vo (phong.vodinh@gmail.com, dinphong.vo@cea.fr)

      SVM-lin: Fast SVM Solvers for Supervised and Semi-supervised Learning

      This file is part of SVM-lin.

      SVM-lin is free software; you can redistribute it and/or modify
      it under the terms of the GNU General Public License as published by
      the Free Software Foundation; either version 2 of the License, or
      (at your option) any later version.

      SVM-lin is distributed in the hope that it will be useful,
      but WITHOUT ANY WARRANTY; without even the implied warranty of
      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
      GNU General Public License for more details.

      You should have received a copy of the GNU General Public License
      along with SVM-lin (see gpl.txt); if not, write to the Free Software
      Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
'''
import os, sys 
import numpy as np 
from scipy.sparse import csr_matrix
from sklearn.datasets import load_svmlight_file
from ctypes import c_double
from ssl import * 

def ssl_transductive_train(Xl, Xn, Xu, verbose, **kwargs):
	if not (isinstance(Xl, csr_matrix) or isinstance(Xl, np.ndarray)) or not ( isinstance(Xn, csr_matrix) or isinstance(Xn, np.ndarray) ) or not ( isinstance(Xu, csr_matrix) or isinstance(Xu, np.ndarray) ):
		raise ValueError('input data type must be either csr_matrix or ndarray')

	if isinstance(Xl, np.ndarray):
		Xl = csr_matrix(Xl)

	if isinstance(Xn, np.ndarray):
		Xn = csr_matrix(Xn)

	if isinstance(Xu, np.ndarray):
		Xu = csr_matrix(Xu)

	if Xl.shape[1] != Xu.shape[1] or Xl.shape[1] != Xn.shape[1] or Xu.shape[1] != Xn.shape[1]:
		raise ValueError('all input data must have the same dimension')

	# prepare Data, Options, Weights, and Outputs
	ssl_data = data()
	ssl_data.from_data_chunks(Xl, Xn, Xu)
	ssl_weights = vector_double()
	ssl_outputs = vector_double()
	ssl_options = options(**kwargs)

	libssl.ssl_train(ssl_data, ssl_options, ssl_weights, ssl_outputs, verbose)

	clf = np.array(np.fromiter(ssl_weights.vec, dtype=float, count=ssl_weights.d))

	#libssl.clear_data(ssl_data)
	libssl.clear_vec_double(ssl_outputs)
	libssl.clear_vec_double(ssl_weights)

	return clf


def ssl_train_with_data(X, y, verbose, **kwargs):
	# check y
	if not isinstance(y, np.ndarray):
		if not isinstance(y, list):
			raise ValueError('y must be an iterable type (list, numpy.ndarray)')
		else:
			y = np.array(y)
	else:
		if np.prod(y.shape) != y.shape[0]:
			raise ValueError('y must be a column or row vector')

	# check y
	labels = set(y)
	if not (labels == set([1.0,-1.0,0.0])) and not (labels == set([1.0,-1.0])):
		raise ValueError('label array must contain positive(+1) and negative (-1) samples, and optionally unlabeled ones (0).')

	# check x vs. y
	if not isinstance(X, np.ndarray) and not isinstance(X, csr_matrix):
		raise ValueError('X and y must be either numpy.ndarray or scipy.sparse.csr_matrix')
	elif X.shape[0] != y.shape[0]:
		raise ValueError('X and y must have  the same number of samples')

	if isinstance(X, np.ndarray):
		X = csr_matrix(X)

	ssl_data = data()
	#ssl_data.clone_data_no_bias(X,y)
	ssl_data.from_data(X,y)
	ssl_weights = vector_double()
	ssl_options = options(**kwargs)
	ssl_outputs = vector_double()

	libssl.ssl_train(ssl_data, ssl_options, ssl_weights, ssl_outputs, verbose)
	
	clf = np.array(np.fromiter(ssl_weights.vec, dtype=np.float64, count=ssl_weights.d))

	#libssl.clear_data(ssl_data)
	libssl.clear_vec_double(ssl_outputs)
	libssl.clear_vec_double(ssl_weights)

	return clf


def ssl_evaluate_online(X, y, w, verbose=True):
	if not isinstance(y, np.ndarray):
		if not isinstance(y, list):
			raise ValueError('y must be an iterable type (list, numpy.ndarray)')
		else:
			y = np.array(y)
	else:
		if np.prod(y.shape) != y.shape[0]:
			raise ValueError('y must be a column or row vector')

	# check y
	labels = set(y)
	if not (labels == set([1.0,-1.0,0.0])) and not (labels == set([1.0,-1.0])):
		raise ValueError('label array must contain positive(+1) and negative (-1) samples, and optionally unlabeled ones (0).')

	# check x vs. y
	if not isinstance(X, np.ndarray) and not isinstance(X, csr_matrix):
		raise ValueError('X and y must be either numpy.ndarray or scipy.sparse.csr_matrix')
	elif X.shape[0] != y.shape[0]:
		raise ValueError('X and y must have  the same number of samples')

	if isinstance(X, np.ndarray):
		X = csr_matrix(X)

	# check weights w
	if not isinstance(w, np.ndarray):
		if not isinstance(w, list):
			raise ValueError('w must be an iterable type (list, numpy.ndarray)')
		else:
			w = np.array(w)
	else:
		if np.prod(w.shape) != w.shape[0]:
			raise ValueError('w must be a column or row vector')

	# dimensionality consistency with X's
	if X.shape[1] > len(w)-1:
		raise ValueError('w[{}] and X[{}x{}] must reside on the same dimensional space'.format(w.shape[0], X.shape[0], X.shape[1]))

	ssl_data = data()
	ssl_data.from_data(X, y)
	
	# C allocation
	ssl_weights = vector_double()
	libssl.init_vec_double(ssl_weights, len(w), 0.0)
	for i, v in enumerate(w):
		ssl_weights.vec[i] = v
	
	ssl_outputs = vector_double()
	
	ssl_gt = vector_double()
	libssl.init_vec_double(ssl_gt, len(y), 0.0)
	for i, v in enumerate(y):
		ssl_gt.vec[i] = v

	print ''
	libssl.ssl_predict_online(ssl_data, ssl_weights, ssl_outputs)
	acc = libssl.ssl_evaluate(ssl_outputs, ssl_gt, verbose)

	# release memory
	libssl.clear_vec_double(ssl_weights)
	libssl.clear_vec_double(ssl_outputs)
	libssl.clear_vec_double(ssl_gt)

	return acc


def ssl_predict_online(X, w):
	# check x vs. y
	if not isinstance(X, np.ndarray) and not isinstance(X, csr_matrix):
		raise ValueError('X and y must be either numpy.ndarray or scipy.sparse.csr_matrix')
	
	if isinstance(X, np.ndarray):
		X = csr_matrix(X)

	# check weights w
	if not isinstance(w, np.ndarray):
		if not isinstance(w, list):
			raise ValueError('w must be an iterable type (list, numpy.ndarray)')
		else:
			w = np.array(w)
	else:
		if np.prod(w.shape) != w.shape[0]:
			raise ValueError('w must be a column or row vector')

	# dimensionality consistency with X's
	if X.shape[1] > w.shape[0] - 1:
		raise ValueError('w and X must reside on the same dimensional space')

	ssl_data = data()
	ssl_data.from_data(X, np.zeros((X.shape[0],)))
	
	# C allocation
	ssl_weights = vector_double()
	libssl.init_vec_double(ssl_weights, len(w), 0.0)
	for i, v in enumerate(w):
		ssl_weights.vec[i] = v
	
	ssl_outputs = vector_double()
	
	print ''
	libssl.ssl_predict_online(ssl_data, ssl_weights, ssl_outputs)
	
	scores = np.array(np.fromiter(ssl_outputs.vec, dtype=np.float64, count=ssl_outputs.d))

	# release memory
	libssl.clear_vec_double(ssl_weights)
	libssl.clear_vec_double(ssl_outputs)
	
	return scores


def ssl_train(data_file, verbose, **kwargs):
	if not os.path.isfile(data_file):
		raise ValueError('File not found')
	X, y = load_svmlight_file(data_file)
	w = ssl_train_with_data(X, y, verbose, **kwargs)
	return w

def ssl_evaluate(data_file, label_file, model_file, verbose=True):
	if not os.path.isfile(data_file) or not os.path.isfile(model_file) or not os.path.isfile(label_file):
		raise ValueError('File(s) not found')

	# load data, label, and learned model
	w = np.loadtxt(model_file)
	gt = np.loadtxt(label_file)

	ssl_weights = vector_double()
	libssl.init_vec_double(ssl_weights, len(w), 0.0)
	for i, v in enumerate(w):
		ssl_weights.vec[i] = v
	
	ssl_outputs = vector_double()

	ssl_gt = vector_double()
	libssl.init_vec_double(ssl_gt, len(gt), 0.0)
	for i, v in enumerate(gt):
		ssl_gt.vec[i] = v

	# test 
	libssl.ssl_predict(data_file, ssl_weights, ssl_outputs)
	# evaluate
	print ''
	acc = libssl.ssl_evaluate(ssl_outputs, ssl_gt, verbose)
	
	libssl.clear_vec_double(ssl_outputs)

	return acc


##################################################################

# print 'Testing <ssl_train>...'
# w = ssl_train('../example/training_data', True, algo=2, lambda_l=0.001, lambda_u=1)
# np.savetxt('../example/weights', w)
# ssl_evaluate('../example/test_examples', '../example/test_labels', '../example/weights', True)

# print ''
# print 'Testing <ssl_transductive_train>...'
# X, y = load_svmlight_file('../example/training_data')
# X = X.todense()
# Xl = X[y==1.0,:]
# Xn = X[y==-1.0,:]
# Xu = X[y==0.0,:]
# print '{} positive examples'.format(Xl.shape[0])
# print '{} negative examples'.format(Xn.shape[0])
# print '{} unlabeled examples'.format(Xu.shape[0])
# w = ssl_transductive_train(Xl, Xn, Xu, True, algo=2, lambda_l=0.001, lambda_u=0)

# # X, y = load_svmlight_file('../example/test_data', zero_based=False)
# # ssl_evaluate_online(X, y, w, True)

# np.savetxt('../example/weights', w)
# ssl_evaluate('../example/test_examples', '../example/test_labels', '../example/weights', True)

# ### Test read-write data consistency
# X, y = load_svmlight_file('../example/dummy', zero_based=False)
# datum = data()
# datum.from_data(X, y)
# datum.dump('../example/dumped')
