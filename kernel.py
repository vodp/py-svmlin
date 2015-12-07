from ctypes import * 
from ctypes.util import find_library 
from os import path 
import numpy as np 
from scipy.sparse import csr_matrix
import sys

# load library first
try:
	dirname = path.dirname(path.abspath(__file__))
	if sys.platform == 'win32':
		print 'not applicable'
	else:
		libknn = CDLL(path.join(dirname, 'libpsdkernel.so'))
except:
	if find_library('psdkernel'):
		libknn = CDLL(find_library('psdkernel'))
	else: 
		raise Exception('LIBpsdkernel not found!')


def genFields(names, types):
	return list(zip(names, types))		

def fillprototype(f, restype, argtypes):
	f.restype = restype
	f.argtypes = argtypes

class data(Structure):
	_names = ['m', 'd', 'nnz', 'val', 'rowptr', 'colind']
	_types = [c_ulong, c_int, c_ulong, POINTER(c_double), POINTER(c_ulong), POINTER(c_int)]
	_fields_ = genFields(_names, _types)

	def clone_data(self, X):
		self.m = X.shape[0]
		self.d = X.shape[1]
		self.nnz = X.nnz

		self.val = (c_double * self.nnz)()
		for i, v in enumerate(X.data):
			self.val[i] = v

		self.colind = (c_int * self.nnz)()
		for i, v in enumerate(X.indices):
			self.colind[i] = v

		self.rowptr = (c_ulong * (self.m+1))()
		for i, v in enumerate(X.indptr):
			self.rowptr[i] = v

def compute_PSDkernel(X, Z, kern='rbf', sigma=1.0, verbose=False):
	if not isinstance(X, csr_matrix):
		X = csr_matrix(X)
	if not isinstance(Z, csr_matrix):
		Z = csr_matrix(Z)
		
	x = data()
	x.clone_data(X)

	z = data()
	z.clone_data(Z)

	if kern == 'rbf':
		libknn.classify_knn_cosine(mydata, mytest, prediction, knn, verbose)
	elif kern == 'lin':
		libknn.classify_knn_l2(mydata, mytest, prediction, knn, verbose)
	else:
		raise Error('unknown kernel type')

	K = np.array(np.fromiter(k, dtype=float, count=Z.shape[0]))
	return prediction
