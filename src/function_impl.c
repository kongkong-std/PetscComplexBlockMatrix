#include "main.h"

void RealMatrixAssemble(const Mat *mat, Mat *mat_re)
{
    PetscInt row_loc = 0, col_loc = 0;
    PetscCall(MatGetLocalSize(*mat, &row_loc, &col_loc));

    PetscBool done = PETSC_TRUE;
    const PetscInt *csr_ia = NULL;
    const PetscInt *csr_ja = NULL;
    PetscInt loc_row_start = 0, loc_row_end = 0;
    PetscCall(MatGetRowIJ(*mat, 0, PETSC_FALSE, PETSC_TRUE, NULL, &csr_ia, &csr_ja, &done));
    PetscCall(MatGetOwnershipRange(*mat, &loc_row_start, &loc_row_end));

printf("loc nnz = %d, loc_row_start = %d, loc_row_end = %d\n", csr_ia[row_loc], loc_row_start, loc_row_end);

    PetscCall(MatCreate(PETSC_COMM_WORLD, mat_re));
    PetscCall(MatSetSizes(*mat_re, row_loc, col_loc, PETSC_DETERMINE, PETSC_DETERMINE));
    PetscCall(MatSetType(*mat_re, MATAIJ));
    PetscCall(MatSetUp(*mat_re));

    for (int index = loc_row_start; index < loc_row_end; ++index)
    {
        int index_start = csr_ia[index];
        int index_end = csr_ia[index + 1];
        for (int index_j = index_start; index_j < index_end; ++index_j)
        {
            PetscScalar val_tmp;
            PetscCall(MatGetValue(*mat, index, csr_ja[index_j], &val_tmp));

            PetscScalar val_tmp_re = PetscRealPart(val_tmp);
            PetscCall(MatSetValues(*mat_re, 1, &index, 1, csr_ja + index_j, &val_tmp_re, INSERT_VALUES));
        }
    }

    PetscCall(MatAssemblyBegin(*mat_re, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(*mat_re, MAT_FINAL_ASSEMBLY));
}

void ImaginaryMatrixAssemble(const Mat *mat, Mat *mat_im)
{
    PetscInt row_loc = 0, col_loc = 0;
    PetscCall(MatGetLocalSize(*mat, &row_loc, &col_loc));

    PetscBool done = PETSC_TRUE;
    const PetscInt *csr_ia = NULL;
    const PetscInt *csr_ja = NULL;
    PetscInt loc_row_start = 0, loc_row_end = 0;
    PetscCall(MatGetRowIJ(*mat, 0, PETSC_FALSE, PETSC_TRUE, NULL, &csr_ia, &csr_ja, &done));
    PetscCall(MatGetOwnershipRange(*mat, &loc_row_start, &loc_row_end));

    PetscCall(MatCreate(PETSC_COMM_WORLD, mat_im));
    PetscCall(MatSetSizes(*mat_im, row_loc, col_loc, PETSC_DETERMINE, PETSC_DETERMINE));
    PetscCall(MatSetType(*mat_im, MATAIJ));
    PetscCall(MatSetUp(*mat_im));

    for (int index = loc_row_start; index < loc_row_end; ++index)
    {
        int index_start = csr_ia[index];
        int index_end = csr_ia[index + 1];
        for (int index_j = index_start; index_j < index_end; ++index_j)
        {
            PetscScalar val_tmp;
            PetscCall(MatGetValue(*mat, index, csr_ja[index_j], &val_tmp));

            PetscScalar val_tmp_re = PetscImaginaryPart(val_tmp);
            PetscCall(MatSetValues(*mat_im, 1, &index, 1, csr_ja + index_j, &val_tmp_re, INSERT_VALUES));
        }
    }

    PetscCall(MatAssemblyBegin(*mat_im, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(*mat_im, MAT_FINAL_ASSEMBLY));
}

void MatrixAssemble(const CSRMatrix *mat_a, Mat *mat)
{
    int n = mat_a->n;
    PetscCall(MatCreate(PETSC_COMM_WORLD, mat));
    PetscCall(MatSetSizes(*mat, PETSC_DECIDE, PETSC_DECIDE, n, n));
    PetscCall(MatSetType(*mat, MATAIJ));
    PetscCall(MatSetUp(*mat));

    PetscInt loc_row_start = 0, loc_row_end = 0;
    PetscCall(MatGetOwnershipRange(*mat, &loc_row_start, &loc_row_end));

    for (int index = loc_row_start; index < loc_row_end; ++index)
    {
        int index_start = mat_a->row_ptr[index];
        int index_end = mat_a->row_ptr[index + 1];
        for (int index_j = index_start; index_j < index_end; ++index_j)
        {
            PetscScalar val_tmp;
            val_tmp = mat_a->val_re[index_j] + mat_a->val_im[index_j] * PETSC_i;
            PetscCall(MatSetValues(*mat, 1, &index, 1, mat_a->col_idx + index_j, &val_tmp, INSERT_VALUES));
        }
    }

    PetscCall(MatAssemblyBegin(*mat, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(*mat, MAT_FINAL_ASSEMBLY));
}

void VectorAssemble(const CSRVector *vec_v, Vec *vec)
{
    int n = vec_v->n;
    PetscCall(VecCreate(PETSC_COMM_WORLD, vec));
    PetscCall(VecSetSizes(*vec, PETSC_DECIDE, n));
    PetscCall(VecSetType(*vec, VECMPI));

    PetscInt loc_row_start = 0, loc_row_end = 0;
    PetscCall(VecGetOwnershipRange(*vec, &loc_row_start, &loc_row_end));

    for (int index = loc_row_start; index < loc_row_end; ++index)
    {
        PetscScalar val_tmp;
        val_tmp = vec_v->val_re[index] + vec_v->val_im[index] * PETSC_i;
        PetscCall(VecSetValues(*vec, 1, &index, &val_tmp, INSERT_VALUES));
    }

    PetscCall(VecAssemblyBegin(*vec));
    PetscCall(VecAssemblyEnd(*vec));
}