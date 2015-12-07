#include "kernel.h"

extern "C" struct sparse_matrix **create_sparse_matrix()
{
	struct sparse_matrix *mx = (sparse_matrix*) malloc(sizeof(sparse_matrix))
	mx->m = 0;
	mx->d = 0;
	mx->nnz = 0;
	mx->val = 0;
	mx->rowptr = 0;
	mx->colind = 0;

	return &mx;
}

extern "C" void delete_sparse_matrix(struct sparse_matrix* matrix)
{
	free(matrix->val);
	free(matrix->rowptr);
	free(matrix->colind);
	free(matrix);
}

extern "C" struct sparse_matrix **compute_rbf(const struct sparse_matrix *X, const struct sparse_matrix *Z, float sigma)
{
	unsigned long int i, j, k, l;
	unsigned long int aa, ab, ba, bb;
	double l2, tmp;

	double_vec_t val;
	int_vec_t rowptr;
	int_vec_t colind;
	
	//#pragma omp parallel for private(j, l2, tmp, Q, aa, ab, ba, bb, k, l) shared(X, Z, labels)
	for (i = 0; i < Z->m; ++i)
	{
		aa = Z->rowptr[i];
		ab = Z->rowptr[i+1];

		// tmp_data = (double*) malloc(sizeof(double)*(ab - aa));
		// memcpy(tmp_data, Z->val+aa, sizeof(double)*(ab - aa));
		
		// #pragma omp parallel for private(l2, tmp, ba, bb, k, l) shared(X, tmp_data, labels, aa, ab, Q)
		unsigned long int offset = 0;
		rowptr.push_back(offset);
		for(j = 0; j < X->m; ++j)
		{
			ba = X->rowptr[j];
			bb = X->rowptr[j+1];

			k = aa;
			l = ba;
			l2 = 0;
			while (k < ab && l < bb)
			{
				if (Z->colind[k] == X->colind[l])
				{
					tmp = (Z->val[k] - X->val[l]);
					l2 += tmp*tmp;
					k++;
					l++;
				}
				else if (Z->colind[k] > X->colind[l])
				{
					tmp = X->val[l];
					l2 += tmp*tmp;
					l++;
				}
				else
				{
					tmp = Z->val[k];
					l2 += tmp*tmp;
					k++;
				}
			}

			if (l2 > 0)
			{
				val.push_back(exp(-l2/(2.0*sigma)));
				colind.push_back(j);
				offset++;
			}
		}

		rowptr.push_back(offset);
	}

	// re-format the kernel result into a sparse matrix
	struct sparse_matrix *Z = (sparse_matrix*) malloc(sizeof(sparse_matrix));
	Z->m = rowptr.size()-1;
	Z->d = rowptr.size()-1;
	Z->nnz = val.size();
	Z->val = (double*) malloc(sizeof(double)*val.size());
	Z->rowptr = (unsigned long int*) malloc(sizeof(unsigned long int)*rowptr.size());
	Z->colind = (int*) malloc(sizeof(int)*colind.size());

	std:copy(val.begin(), val.end(), Z->val);
	std:copy(colind.begin(), colind.end(), Z->colind);
	std:copy(rowptr.begin(), rowptr.end(), Z->rowptr);

	return &Z;
}

extern "C" void compute_linear(const struct sparse_matrix *X, const struct sparse_matrix *Z)
{
	unsigned long int i, j, k, l;
	unsigned long int aa, ab, ba, bb;
	double l2, tmp;

	double_vec_t val;
	int_vec_t rowptr;
	int_vec_t colind;
	
	//#pragma omp parallel for private(j, l2, tmp, Q, aa, ab, ba, bb, k, l) shared(X, Z, labels)
	for (i = 0; i < Z->m; ++i)
	{
		aa = Z->rowptr[i];
		ab = Z->rowptr[i+1];

		// tmp_data = (double*) malloc(sizeof(double)*(ab - aa));
		// memcpy(tmp_data, Z->val+aa, sizeof(double)*(ab - aa));
		
		// #pragma omp parallel for private(l2, tmp, ba, bb, k, l) shared(X, tmp_data, labels, aa, ab, Q)
		unsigned long int offset = 0;
		rowptr.push_back(offset);
		for(j = 0; j < X->m; ++j)
		{
			ba = X->rowptr[j];
			bb = X->rowptr[j+1];

			k = aa;
			l = ba;
			l2 = 0;
			while (k < ab && l < bb)
			{
				if (Z->colind[k] == X->colind[l])
				{
					l2 += Z->val[k] * X->val[l];
					k++;
					l++;
				}
				else if (Z->colind[k] > X->colind[l])
					l++;
				else
					k++;
			}

			if (l2 > 0)
			{
				val.push_back(l2);
				colind.push_back(j);
				offset++;
			}
		}

		rowptr.push_back(offset);
	}

	// re-format the kernel result into a sparse matrix
	struct sparse_matrix *Z = (sparse_matrix*) malloc(sizeof(sparse_matrix));
	Z->m = rowptr.size()-1;
	Z->d = rowptr.size()-1;
	Z->nnz = val.size();
	Z->val = (double*) malloc(sizeof(double)*val.size());
	Z->rowptr = (unsigned long int*) malloc(sizeof(unsigned long int)*rowptr.size());
	Z->colind = (int*) malloc(sizeof(int)*colind.size());

	std:copy(val.begin(), val.end(), Z->val);
	std:copy(colind.begin(), colind.end(), Z->colind);
	std:copy(rowptr.begin(), rowptr.end(), Z->rowptr);

	return &Z;
}
