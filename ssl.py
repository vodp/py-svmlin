'''   Copyright  2014 Phong Vo (phong.vodinh@gmail.com, dinphong.vo@cea.fr)

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
from ctypes import * 
from ctypes.util import find_library
from os import path
import numpy as np 
from scipy.sparse import csr_matrix
import sys 

__all__ = ['libssl', 'data', 'vector_double', 'vector_int', 'options', 'RLS', 'SVM', 'TSVM', 'DA_SVM', 'c_int']

# load library 
try: 
	dirname = path.dirname(path.abspath(__file__))
	if sys.platform == 'win32':
		print 'not implemented yet!'
	else:
		#libssl = CDLL(path.join(dirname, 'libtsvm.so'))
		libssl = CDLL(path.join('/home/phong/cea/cea/rerank/libtsvm.so'))
		print 'libtsvm library loaded.'
except:
	if find_library('tsvm'):
		libssl = CDLL(find_library('tsvm'))
	else:
		raise Exception('LIBTSVM library not found')

RLS = 0
SVM = 1
TSVM = 2
DA_SVM = 3

def genFields(names, types):
	return list(zip(names, types))

def fillprototype(f, restype, argtypes):
	f.restype = restype
	f.argtypes = argtypes

# construct constants
class data(Structure):
	'''
	m: number of examples
	l: number of labeled examples
	u: number of unlabeled examples
	n: number of features
	nz: number of non-zeros
	val: data values (CRS format)
	rowptr: n+1 vector (CRS format)
	colind: nz elements (CRS format)
	Y: labels
	C: cost associated with each examples
	'''
	_names = ['m', 'l', 'u', 'n', 'nz', 'val', 'rowptr', 'colind', 'Y', 'C']
	_types = [c_int, c_int, c_int, c_int, c_int, POINTER(c_double), POINTER(c_int), POINTER(c_int), POINTER(c_double), POINTER(c_double) ]
	_fields_ = genFields(_names, _types)

	def __str__(self):
		s = ''
		attrs = data._names + list(self.__dict__.keys())
		values = map(lambda attr: getattr(self, attr), attrs)
		for attr, val in zip(attrs, values):
			s += ('%s: %s\n' % (attr, val))
		s.strip()

		return s

	def from_data_chunks(self, Xl, Xn, Xu):
		self.__frombuffer__ = False
		self.__createfrom__ = 'python'
		
		self.l = Xl.shape[0] + Xn.shape[0]
		self.u = Xu.shape[0]
		self.m = self.l + self.u
		self.n = Xl.shape[1]
		# non-zeros = non-zeros + number of bias
		self.nz = (Xl.nnz + Xn.nnz + Xu.nnz) + self.m

		# set val, colind, and rowptr
		self.val = (c_double * self.nz)()
		self.colind = (c_int * self.nz)()
		self.rowptr = (c_int * (self.m + 1))()
		i = 0
		t = 0
		# the variable rowptr will be accumulated during data copy
		# the variables indptr and data are copied by chunks
		for j in range(len(Xl.indptr)-1):
			self.rowptr[t] = i
			values = Xl.data[Xl.indptr[j]:Xl.indptr[j+1]]
			indices = Xl.indices[Xl.indptr[j]:Xl.indptr[j+1]]
			for v, ix in zip(values, indices): 
				self.val[i] = v
				self.colind[i] = ix
				i += 1
			# add bias 1
			self.val[i] = 1.0
			self.colind[i] = self.n
			i += 1
			t += 1

		for j in range(len(Xn.indptr)-1):
			self.rowptr[t] = i
			values = Xn.data[Xn.indptr[j]:Xn.indptr[j+1]]
			indices = Xn.indices[Xn.indptr[j]:Xn.indptr[j+1]]
			for v, ix in zip(values, indices): 
				self.val[i] = v
				self.colind[i] = ix
				i += 1
			# add bias 1
			self.val[i] = 1.0
			self.colind[i] = self.n
			i += 1
			t += 1

		for j in range(len(Xu.indptr)-1):
			self.rowptr[t] = i
			values = Xu.data[Xu.indptr[j]:Xu.indptr[j+1]]
			indices = Xu.indices[Xu.indptr[j]:Xu.indptr[j+1]]
			for v, ix in zip(values, indices): 
				self.val[i] = v
				self.colind[i] = ix
				i += 1
			# add bias 1
			self.val[i] = 1.0
			self.colind[i] = self.n
			i += 1
			t += 1
		self.rowptr[t] = self.nz
		self.n += 1

		# set labels
		self.Y = (c_double * self.m)()
		i = 0
		while i < Xl.shape[0]:
			self.Y[i] = 1
			i += 1

		while i < self.l:
			self.Y[i] = -1
			i += 1

		while i < self.m:
			self.Y[i] = 0
			i += 1

		# set default cost (1.0) to all examples
		self.C = (c_double * self.m)()
		for i in range(self.m):
			self.C[i] = 1.0


	# def from_data(self, X, y): # no bias of course
	# 	self.__frombuffer__ = True
	# 	self.m = len(y)
	# 	self.l = sum(y != 0)
	# 	self.u = self.m - self.l
	# 	self.n = X.shape[1]
	# 	# non-zeros = non-zeros + number of bias
	# 	self.nz = X.nnz + self.m

	# 	# get a reference  to data pointer of a Numpy object
	# 	X = X.astype(np.float64)
	# 	self.val = (c_double * self.nz)(*X.data)
	# 	self.colind = (c_int * self.nz)(*X.indices)
	# 	self.rowptr = (c_int * (self.m + 1))(*X.indptr)

	# 	# set labels
	# 	y = y.astype(np.float64)
	# 	self.Y = y.ctypes.data_as(POINTER(c_double))
				
	# 	# set default cost (1.0) to all examples
	# 	self.C = (c_double * self.m)()
	# 	for i in range(self.m):
	# 		self.C[i] = 1.0

	# def clone_data_no_bias(self, X, y):
	# 	self.__frombuffer__ = False
	# 	self.m = len(y)
	# 	self.l = sum(y != 0)
	# 	self.u = self.m - self.l
	# 	self.n = X.shape[1]
	# 	# non-zeros = non-zeros + number of bias
	# 	self.nz = X.nnz

	# 	# set val, colind, and rowptr
	# 	self.val = (c_double * self.nz)()
	# 	for i, v in enumerate(X.data):
	# 		self.val[i] = v

	# 	self.colind = (c_int * self.nz)()
	# 	for i, v in enumerate(X.indices):
	# 		self.colind[i] = v

	# 	self.rowptr = (c_int * (self.m + 1))()
	# 	for i, v in enumerate(X.indptr):
	# 		self.rowptr[i] = v

	# 	# set labels
	# 	self.Y = (c_double * self.m)()
	# 	for i,v in enumerate(y):
	# 		self.Y[i] = v
		
	# 	# set default cost (1.0) to all examples
	# 	self.C = (c_double * self.m)()
	# 	for i in range(self.m):
	# 		self.C[i] = 1.0

	def from_data(self, X, y):
		self.__frombuffer__ = False
		# TODO

		# set constants
		self.m = len(y)
		self.l = sum(y != 0)
		self.u = self.m - self.l
		self.n = X.shape[1]
		self.nz = X.nnz + self.m

		# allocate memory
		self.val = (c_double * self.nz)()
		self.colind = (c_int * self.nz)()
		self.rowptr = (c_int * (self.m + 1))()
		self.Y = (c_double * self.m)()
		self.C = (c_double * self.m)()
		
		# copying data
		i = 0
		for j in range(len(X.indptr)-1):
			self.rowptr[j] = i
			values = X.data[X.indptr[j]:X.indptr[j+1]]
			indices = X.indices[X.indptr[j]:X.indptr[j+1]]
			for v, ix in zip(values, indices): 
				self.val[i] = v
				self.colind[i] = ix
				i += 1
			# add bias 1
			self.val[i] = 1.0
			self.colind[i] = self.n
			i += 1
		self.rowptr[j+1] = self.nz

		#for i in range(j):
		#	self.colind[self.rowptr[i+1]-1] = self.n
		self.n += 1		

		# set labels
		for i,v in enumerate(y):
			self.Y[i] = v
		
		# set default cost (1.0) to all examples
		for i in range(self.m):
			self.C[i] = 1.0

	def __init__(self):
		self.__createfrom__ = 'python'
		self.__frombuffer__ = True


	def dump(self, filename):
		with open(filename, 'wt') as fout:
			for j in range(self.m):
				# write label
				fout.write('%d\t' % (self.Y[j]))

				# write non-zero indices
				start_ix = self.rowptr[j]
				stop_ix = self.rowptr[j+1]
				for i in range(start_ix, stop_ix-1):
					fout.write('%d:%2.4f ' % (self.colind[i]+1, self.val[i]))
				fout.write('\n')
				
	# def __del__(self):
	# 	if hasattr(self, '__frombuffer__') and self.__frombuffer__ == False:
	# 		libssl.clear_vec_double(self.val)
	# 		libssl.clear_vec_double(self.colind)
	# 		libssl.clear_vec_double(self.rowptr)
	# 		libssl.clear_vec_double(self.C)
	# 		libssl.clear_vec_double(self.Y)


class vector_double(Structure):
	_names = ['d', 'vec']
	_types = [c_int, POINTER(c_double)]
	_fields_ = genFields(_names, _types)

	def __init__(self):
		self.__createfrom__ = 'python'


class vector_int(Structure):
	_names = ['d', 'vec']			
	_types = [c_int, POINTER(c_int)]
	_fields_ = genFields(_names, _types)

	def __init__(self):
		self.__createfrom__ = 'python'


class options(Structure):
	_names = ['algo', 'lambda_l', 'lambda_u', 'S', 'R', 'Cp', 'Cn', 'epsilon', 'cgitermax', 'mfnitermax']
	_types = [c_int, c_double, c_double, c_int, c_double, c_double, c_double, c_double, c_int, c_int]
	_fields_ = genFields(_names, _types)

	def __init__(self, **kwargs):
		self.set_defaults()

		if kwargs:
			if 'algo' in kwargs.keys():
				self.algo = kwargs['algo']

			if 'lambda_l' in kwargs.keys():
				self.lambda_l = kwargs['lambda_l']

			if 'lambda_u' in kwargs.keys():
				self.lambda_u = kwargs['lambda_u']

			if 'S' in kwargs.keys():
				self.S = kwargs['S']

			if 'R' in kwargs.keys():
				self.R = kwargs['R']

			if 'Cp' in kwargs.keys():
				self.Cp = kwargs['Cp']

			if 'Cn' in kwargs.keys():
				self.Cn = kwargs['Cn']

			if 'epsilon' in kwargs.keys():
				self.epsilon = kwargs['epsilon']

			if 'cgitermax' in kwargs.keys():
				self.cgitermax = kwargs['cgitermax']

			if 'mfnitermax' in kwargs.keys():
				self.mfnitermax = kwargs['mfnitermax']

	def set_defaults(self):
		self.algo = RLS
		self.lambda_l = 1.0
		self.lambda_u = 1.0
		self.S = 10000
		self.R = 0.5 
		self.Cp = 1.0 
		self.Cn = 1.0 
		self.epsilon = 1e-6
		self.cgitermax = 10000
		self.mfnitermax = 50

	def __str__(self):
		s = ''
		attrs = options._names + list(self.__dict__.keys())
		values = map(lambda attr: getattr(self, attr), attrs)
		for attr, val in zip(attrs, values):
			s += ('%s: %s\n'%(attr, val))
		s.strip()
		return s


fillprototype(libssl.ssl_train, None, [POINTER(data), POINTER(options), POINTER(vector_double), POINTER(vector_double), c_int])
fillprototype(libssl.ssl_predict, None, [c_char_p, POINTER(vector_double), POINTER(vector_double)])
fillprototype(libssl.ssl_predict_online, None, [POINTER(data), POINTER(vector_double), POINTER(vector_double)])
fillprototype(libssl.ssl_evaluate, None, [POINTER(vector_double), POINTER(vector_double), c_int])
fillprototype(libssl.clear_data, None, [POINTER(data)])
fillprototype(libssl.clear_vec_double, None, [POINTER(vector_double)])
fillprototype(libssl.clear_vec_int, None, [POINTER(vector_int)])
fillprototype(libssl.init_vec_double, None, [POINTER(vector_double), c_int, c_double])
fillprototype(libssl.init_vec_int, None, [POINTER(vector_int), c_int])
# fillprototype(libssl.SetData, None, [POINTER(data), c_int, c_int, c_int, c_int, c_int, POINTER(c_double), POINTER(c_int), POINTER(c_int), POINTER(c_double), POINTER(c_double)])
# fillprototype(libssl.GetLabeledData, None, [POINTER(data), POINTER(data)])
# fillprototype(libssl.norm_square, c_double, [POINTER(vector_double)])
