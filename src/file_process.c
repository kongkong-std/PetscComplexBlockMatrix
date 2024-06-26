#include "file_process.h"

void CSRMatrixFileProcess(const char *path, CSRMatrix *mat)
{
    FILE *fp = NULL;
    if ((fp = fopen(path, "rb")) == NULL)
    {
        fprintf(stderr, "Cannot open matrix file - \'%s\'\n", path);
        exit(EXIT_FAILURE);
    }

    int n = 0, nnz = 0;
    fscanf(fp, "%d%d", &n, &nnz);

    mat->n = n;
    mat->nnz = nnz;
    if ((mat->row_idx = (int *)malloc(n * sizeof(int))) == NULL ||
        (mat->row_ptr = (int *)malloc((n + 1) * sizeof(int))) == NULL ||
        (mat->col_idx = (int *)malloc(nnz * sizeof(int))) == NULL ||
        (mat->val_re = (double *)malloc(nnz * sizeof(double))) == NULL ||
        (mat->val_im = (double *)malloc(nnz * sizeof(double))) == NULL)
    {
        fprintf(stderr, "Memory allocation failed - matrix data\n");
        exit(EXIT_FAILURE);
    }

    for (int index = 0; index < n; ++index)
    {
        fscanf(fp, "%d", mat->row_idx + index);
    }
    for (int index = 0; index < n + 1; ++index)
    {
        fscanf(fp, "%d", mat->row_ptr + index);
    }
    for (int index = 0; index < nnz; ++index)
    {
        fscanf(fp, "%d", mat->col_idx + index);
    }
    for (int index = 0; index < nnz; ++index)
    {
        fscanf(fp, "%lf%lf", mat->val_re + index, mat->val_im + index);
    }

    fclose(fp);
}

void CSRVectorFileProcess(const char *path, CSRVector *vec)
{
    FILE *fp = NULL;
    if ((fp = fopen(path, "rb")) == NULL)
    {
        fprintf(stderr, "Cannot open vector file - \'%s\'\n", path);
        exit(EXIT_FAILURE);
    }

    int n = 0;
    fscanf(fp, "%d", &n);
    vec->n = n;

    if ((vec->row_idx = (int *)malloc(n * sizeof(int))) == NULL ||
        (vec->val_re = (double *)malloc(n * sizeof(double))) == NULL ||
        (vec->val_im = (double *)malloc(n * sizeof(double))) == NULL)
    {
        fprintf(stderr, "Memory allocation failed - vector data\n");
        exit(EXIT_FAILURE);
    }

    for (int index = 0; index < n; ++index)
    {
        fscanf(fp, "%d%lf%lf", vec->row_idx + index, vec->val_re + index, vec->val_im + index);
    }

    fclose(fp);
}

void CSRMatrixFree(CSRMatrix *mat)
{
    free(mat->row_idx);
    free(mat->row_ptr);
    free(mat->col_idx);
    free(mat->val_re);
    free(mat->val_im);
}

void CSRVectorFree(CSRVector *vec)
{
    free(vec->row_idx);
    free(vec->val_re);
    free(vec->val_im);
}