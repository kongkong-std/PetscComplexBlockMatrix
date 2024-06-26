/*
 * test petsc with complex block matrix, serial version
 */

#include "main.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    PetscFunctionBeginUser;
    PetscCall(PetscInitialize(&argc, &argv, (char *)0, NULL));

    int irank, nproc;
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &irank);

    char path_mat[PETSC_MAX_PATH_LEN], path_rhs[PETSC_MAX_PATH_LEN], path_sol[PETSC_MAX_PATH_LEN];
    PetscBool flag_path;

    PetscCall(PetscOptionsGetString(NULL, NULL, "-path_mat", path_mat, sizeof(path_mat), &flag_path));
    if (flag_path)
    {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "path to matrix file: %s\n", path_mat));
    }
    else
    {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "specified path to matrix file not found!\n"));
        exit(EXIT_FAILURE);
    }

    PetscCall(PetscOptionsGetString(NULL, NULL, "-path_rhs", path_rhs, sizeof(path_rhs), &flag_path));
    if (flag_path)
    {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "path to rhs file: %s\n", path_rhs));
    }
    else
    {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "specified path to rhs file not found!\n"));
        exit(EXIT_FAILURE);
    }

    PetscCall(PetscOptionsGetString(NULL, NULL, "-path_sol", path_sol, sizeof(path_sol), &flag_path));
    if (flag_path)
    {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "path to solution file: %s\n", path_sol));
    }
    else
    {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "specified path to solution file not found!\n"));
        exit(EXIT_FAILURE);
    }

    CSRMatrix mat_a;
    CSRVector vec_b, vec_x; // rhs and solution

    PetscCall(PetscPrintf(PETSC_COMM_WORLD, ">>>> begin to process linear system file >>>>\n"));
    CSRMatrixFileProcess(path_mat, &mat_a);
    CSRVectorFileProcess(path_rhs, &vec_b);
    CSRVectorFileProcess(path_sol, &vec_x);
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, ">>>> linear system file has been processed !!!\n"));

    Mat solver_a;
    Vec solver_b, solver_x, solver_u; // rhs, solution ,numerical solution

    PetscCall(PetscPrintf(PETSC_COMM_WORLD, ">>>> begin to assemble petsc linear system >>>>\n"));
    MatrixAssemble(&mat_a, &solver_a);
    VectorAssemble(&vec_b, &solver_b);
    VectorAssemble(&vec_x, &solver_x);
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, ">>>> petsc linear system has been assembled !!!\n"));

    PetscCall(VecDuplicate(solver_x, &solver_u));

#if 1
    Vec solver_r;
    PetscReal b_norm_2 = 0.;
    PetscReal r_norm_2 = 0.;

    PetscCall(VecDuplicate(solver_x, &solver_r));

    PetscCall(VecNorm(solver_b, NORM_2, &b_norm_2));

    PetscCall(MatMult(solver_a, solver_x, solver_r));
    PetscCall(VecAXPY(solver_r, -1., solver_b));
    PetscCall(VecNorm(solver_r, NORM_2, &r_norm_2));

    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "            || b ||_2 = %021.16le\n", b_norm_2));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "|| r ||_2 / || b ||_2 = %021.16le\n", r_norm_2 / b_norm_2));
#endif // residual norm

#if 0
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n==== matrix data:\n"));
    PetscCall(MatView(solver_a, PETSC_VIEWER_STDOUT_WORLD));

    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n==== rhs data:\n"));
    PetscCall(VecView(solver_b, PETSC_VIEWER_STDOUT_WORLD));

    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n==== solution data:\n"));
    PetscCall(VecView(solver_x, PETSC_VIEWER_STDOUT_WORLD));

    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n==== numerical solution data:\n"));
    PetscCall(VecView(solver_u, PETSC_VIEWER_STDOUT_WORLD));
#endif // view matrix, vector

    // testing block linear system
    /*
     * solver_a * solver_x = solver_b
     *     solver_a = solver_a_re + solver_a_im * i
     *     solver_x = solver_x_re + solver_x_im * i
     *     solver_b = solver_b_re + solver_b_im * i
     *
     * [solver_a_re    -solver_a_im] [solver_x_re] = [solver_b_re]
     * [solver_a_im     solver_a_re] [solver_x_im] = [solver_b_im]
     */
    Mat solver_a_re, solver_a_im, solver_a_im_oppo; // real part, imaginary part, opposite imaginary part

    RealMatrixAssemble(&solver_a, &solver_a_re);
    ImaginaryMatrixAssemble(&solver_a, &solver_a_im);

    PetscCall(MatDuplicate(solver_a_im, MAT_DO_NOT_COPY_VALUES, &solver_a_im_oppo));
    PetscCall(MatAXPY(solver_a_im_oppo, -1., solver_a_im, SAME_NONZERO_PATTERN));

#if 0
    PetscInt n_re = 0, n_im = 0;
    PetscCall(MatGetSize(solver_a_re, &n_re, NULL));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n==== real part matrix data:\n"));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "size of real part matrix: %d\n", n_re));
    PetscCall(MatView(solver_a_re, PETSC_VIEWER_STDOUT_WORLD));

    PetscCall(MatGetSize(solver_a_im, &n_im, NULL));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n==== imaginary part matrix data:\n"));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "size of imaginary part matrix: %d\n", n_im));
    PetscCall(MatView(solver_a_im, PETSC_VIEWER_STDOUT_WORLD));

    PetscCall(MatGetSize(solver_a_im_oppo, &n_im, NULL));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n==== imaginary part matrix data:\n"));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "size of imaginary part matrix: %d\n", n_im));
    PetscCall(MatView(solver_a_im_oppo, PETSC_VIEWER_STDOUT_WORLD));
#endif // view real/imaginary part matrix

    Mat mat_array[4] = {solver_a_re, solver_a_im_oppo, solver_a_im, solver_a_re};
    Mat solver_block_a;

    PetscCall(MatCreateNest(PETSC_COMM_WORLD, 2, NULL, 2, NULL, mat_array, &solver_block_a));

#if 0
    PetscInt n_block = 0;
    PetscCall(MatGetSize(solver_block_a, &n_block, NULL));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n==== 2x2 block matrix data:\n"));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "size of 2x2 block matrix: %d\n", n_block));
    PetscCall(MatView(solver_block_a, PETSC_VIEWER_STDOUT_WORLD));
#endif // view block matrix

    Vec solver_x_re, solver_x_im;
    Vec solver_b_re, solver_b_im;

    RealVectorAssemble(&solver_b, &solver_b_re);
    ImaginaryVectorAssemble(&solver_b, &solver_b_im);
    RealVectorAssemble(&solver_x, &solver_x_re);
    ImaginaryVectorAssemble(&solver_x, &solver_x_im);

#if 0
    PetscInt n_vec = 0;
    PetscCall(VecGetSize(solver_b_re, &n_vec));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n==== real part rhs data:\n"));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "size of real part of rhs: %d\n", n_vec));
    PetscCall(VecView(solver_b_re, PETSC_VIEWER_STDOUT_WORLD));

    PetscCall(VecGetSize(solver_b_im, &n_vec));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n==== imaginary part rhs data:\n"));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "size of imaginary part of rhs: %d\n", n_vec));
    PetscCall(VecView(solver_b_im, PETSC_VIEWER_STDOUT_WORLD));

    PetscCall(VecGetSize(solver_x_re, &n_vec));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n==== real part solution data:\n"));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "size of real part of solution: %d\n", n_vec));
    PetscCall(VecView(solver_x_re, PETSC_VIEWER_STDOUT_WORLD));

    PetscCall(VecGetSize(solver_x_im, &n_vec));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n==== imaginary part solution data:\n"));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "size of imaginary part of solution: %d\n", n_vec));
    PetscCall(VecView(solver_x_im, PETSC_VIEWER_STDOUT_WORLD));
#endif // view real/imaginar part vector

    Vec rhs_vec_array[2] = {solver_b_re, solver_b_im};
    Vec solver_block_b;
    Vec sol_vec_array[2] = {solver_x_re, solver_x_im};
    Vec solver_block_x;

    PetscCall(VecCreateNest(PETSC_COMM_WORLD, 2, NULL, rhs_vec_array, &solver_block_b));
    PetscCall(VecCreateNest(PETSC_COMM_WORLD, 2, NULL, sol_vec_array, &solver_block_x));

#if 0
    PetscInt n_block_vec = 0;
    PetscCall(VecGetSize(solver_block_b, &n_block_vec));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n==== block rhs data:\n"));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "size of block rhs: %d\n", n_block_vec));
    PetscCall(VecView(solver_block_b, PETSC_VIEWER_STDOUT_WORLD));

    PetscCall(VecGetSize(solver_block_x, &n_block_vec));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n==== block rhs data:\n"));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "size of block rhs: %d\n", n_block_vec));
    PetscCall(VecView(solver_block_x, PETSC_VIEWER_STDOUT_WORLD));
#endif // view block vector

#if 1
    Vec solver_block_r;
    PetscReal b_block_norm_2 = 0.;
    PetscReal r_block_norm_2 = 0.;

    PetscCall(VecDuplicate(solver_block_x, &solver_block_r));

    PetscCall(VecNorm(solver_block_b, NORM_2, &b_block_norm_2));

    PetscCall(MatMult(solver_block_a, solver_block_x, solver_block_r));
    PetscCall(VecAXPY(solver_block_r, -1., solver_block_b));
    PetscCall(VecNorm(solver_block_r, NORM_2, &r_block_norm_2));

    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "block             || b ||_2 = %021.16le\n", b_block_norm_2));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "block || r ||_2 / || b ||_2 = %021.16le\n", r_block_norm_2 / b_block_norm_2));
#endif // residual norm

#if 0
     // solving linear system with ksp
    KSP ksp;
    PC pc;

    PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
    PetscCall(KSPSetOperators(ksp, petsc_mat_A, petsc_mat_pc));
    PetscCall(KSPGetPC(ksp, &pc));

    PetscCall(KSPSetFromOptions(ksp));
    PetscCall(KSPSolve(ksp, petsc_vec_rhs, petsc_vec_sol));
#endif // ksp option

    // free petsc memory
    PetscCall(MatDestroy(&solver_a));
    PetscCall(MatDestroy(&solver_a_re));
    PetscCall(MatDestroy(&solver_a_im));
    PetscCall(MatDestroy(&solver_a_im_oppo));

    PetscCall(VecDestroy(&solver_b));
    PetscCall(VecDestroy(&solver_x));
    PetscCall(VecDestroy(&solver_u));
    PetscCall(VecDestroy(&solver_b_re));
    PetscCall(VecDestroy(&solver_b_im));
    PetscCall(VecDestroy(&solver_x_re));
    PetscCall(VecDestroy(&solver_x_im));

    // free memory
    CSRMatrixFree(&mat_a);
    CSRVectorFree(&vec_b);
    CSRVectorFree(&vec_x);

    PetscCall(PetscFinalize());
    MPI_Finalize();
    return 0;
}

/*
 * command option
 * -path_mat ../input/csr_mat.txt
 * -path_rhs ../input/rhs_vec.txt
 * -path_sol ../input/sol_vec.txt
 * -ksp_type fgmres -ksp_gmres_restart 1000 -ksp_monitor_true_residual -ksp_max_it 10000
 * -ksp_rtol 1e-6 -pc_type none -user_defined_pc
 */
