#include "main.h"

#ifdef NEW_METHOD
void BlockPetscMatrixAssemble(const Mat *mat_00, const Mat *mat_01,
                              const Mat *mat_10, const Mat *mat_11,
                              Mat *mat)
{
    PetscInt n_mat = 0, m_mat = 0;
    PetscCall(MatGetSize(*mat_00, &n_mat, &m_mat));

    PetscCall(MatCreate(PETSC_COMM_WORLD, mat));
    PetscCall(MatSetSizes(*mat, PETSC_DECIDE, PETSC_DECIDE, 2 * n_mat, 2 * m_mat));
    PetscCall(MatSetType(*mat, MATAIJ));
    PetscCall(MatSetUp(*mat));
}

void BlockPetscVectorAssemble(const Vec *vec_00, const Vec *vec_01,
                              Vec *vec)
{
}
#endif

void CombineTwoVectors(const Vec *vec1, const Vec *vec2, Vec *vec)
{
    PetscInt n1 = 0, n2 = 0;
    PetscInt n = 0;

    PetscCall(VecGetSize(*vec1, &n1));
    PetscCall(VecGetSize(*vec2, &n2));

    n = n1 + n2;
    PetscCall(VecCreate(PETSC_COMM_WORLD, vec));
    PetscCall(VecSetSizes(*vec, PETSC_DECIDE, n));
    PetscCall(VecSetType(*vec, VECMPI));

    Vec vec_array[2] = {*vec1, *vec2};
    PetscCall(VecConcatenate(2, vec_array, vec, NULL));
}

void RealVectorAssemble(const Vec *vec, Vec *vec_re)
{
    PetscInt row_loc = 0;
    PetscCall(VecGetLocalSize(*vec, &row_loc));

    PetscInt loc_row_start = 0, loc_row_end = 0;
    PetscCall(VecGetOwnershipRange(*vec, &loc_row_start, &loc_row_end));

    PetscCall(VecCreate(PETSC_COMM_WORLD, vec_re));
    PetscCall(VecSetSizes(*vec_re, row_loc, PETSC_DETERMINE));
    PetscCall(VecSetUp(*vec_re));

    for (int index = loc_row_start; index < loc_row_end; ++index)
    {
        PetscScalar val_tmp;
        PetscCall(VecGetValues(*vec, 1, &index, &val_tmp));

        PetscScalar val_tmp_re = PetscRealPart(val_tmp);
        PetscCall(VecSetValues(*vec_re, 1, &index, &val_tmp_re, INSERT_VALUES));
    }

    PetscCall(VecAssemblyBegin(*vec_re));
    PetscCall(VecAssemblyEnd(*vec_re));
}

void ImaginaryVectorAssemble(const Vec *vec, Vec *vec_im)
{
    PetscInt row_loc = 0;
    PetscCall(VecGetLocalSize(*vec, &row_loc));

    PetscInt loc_row_start = 0, loc_row_end = 0;
    PetscCall(VecGetOwnershipRange(*vec, &loc_row_start, &loc_row_end));

    PetscCall(VecCreate(PETSC_COMM_WORLD, vec_im));
    PetscCall(VecSetSizes(*vec_im, row_loc, PETSC_DETERMINE));
    PetscCall(VecSetUp(*vec_im));

    for (int index = loc_row_start; index < loc_row_end; ++index)
    {
        PetscScalar val_tmp;
        PetscCall(VecGetValues(*vec, 1, &index, &val_tmp));

        PetscScalar val_tmp_im = PetscImaginaryPart(val_tmp);
        PetscCall(VecSetValues(*vec_im, 1, &index, &val_tmp_im, INSERT_VALUES));
    }

    PetscCall(VecAssemblyBegin(*vec_im));
    PetscCall(VecAssemblyEnd(*vec_im));
}

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

    PetscCall(MatCreate(PETSC_COMM_WORLD, mat_re));
    PetscCall(MatSetSizes(*mat_re, row_loc, col_loc, PETSC_DETERMINE, PETSC_DETERMINE));
    PetscCall(MatSetType(*mat_re, MATAIJ));
    PetscCall(MatSetUp(*mat_re));

    for (int index = loc_row_start; index < loc_row_end; ++index)
    {
        int index_start = csr_ia[index - loc_row_start];
        int index_end = csr_ia[index - loc_row_start + 1];
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
        int index_start = csr_ia[index - loc_row_start];
        int index_end = csr_ia[index - loc_row_start + 1];
        for (int index_j = index_start; index_j < index_end; ++index_j)
        {
            PetscScalar val_tmp;
            PetscCall(MatGetValue(*mat, index, csr_ja[index_j], &val_tmp));

            PetscScalar val_tmp_im = PetscImaginaryPart(val_tmp);
            PetscCall(MatSetValues(*mat_im, 1, &index, 1, csr_ja + index_j, &val_tmp_im, INSERT_VALUES));
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