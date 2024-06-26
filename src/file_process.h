#ifndef FILE_PROCESS_H_
#define FILE_PROCESS_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// struct
typedef struct csr_matrix
{
    /* data */
    int n, nnz;
    int *row_idx, *row_ptr, *col_idx;
    double *val_re, *val_im;
} CSRMatrix;

typedef struct csr_vector
{
    /* data */
    int n;
    int *row_idx;
    double *val_re, *val_im;
} CSRVector;

// function prototype
void CSRMatrixFileProcess(const char * /*path to file*/, CSRMatrix * /*csr matrix*/);
void CSRVectorFileProcess(const char * /*path to file*/, CSRVector * /*csr vector*/);
void CSRMatrixFree(CSRMatrix * /*csr matrix*/);
void CSRVectorFree(CSRVector * /*csr vector*/);

#endif