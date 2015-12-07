#ifndef _kernel_H
#define _kernel_H
#include <math.h>
#include <memory.h>
#include <stdio.h>
#include <float.h>
#include <vector>

typedef std::vector<unsigned long int> int_vec_t;
typedef std::vector<double> double_vec_t;

extern "C" struct sparse_matrix
{
	unsigned long int m;
	int d;
	unsigned long int nnz;
	double *val;
	unsigned long int *rowptr;
	int *colind;
};

extern "C" struct sparse_matrix **create_sparse_matrix();
extern "C" void delete_sparse_matrix(struct sparse_matrix*);

extern "C" struct sparse_matrix **compute_rbf(const struct sparse_matrix *X, const struct sparse_matrix *Z, float sigma);
extern "C" void compute_linear(const struct sparse_matrix *X, const struct sparse_matrix *Z, struct sparse_matrix *K);

// class dynamic_array
// {
// private:
// 	double *val;
// 	unsigned long int *index;
// 	unsigned long int max_len;
// 	unsigned long int pos;

// public:
// 	dynamic_array(void);
// 	dynamic_array(int length);
// 	~dynamic_array(void);

// 	void push(double item);
// 	void clone()
// }