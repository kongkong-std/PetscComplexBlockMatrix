#ifndef MAIN_H_
#define MAIN_H_

#include <petscksp.h>
#include <mpi.h>
#include "file_process.h"

// #define NEW_METHOD

// function prototype
/*
 * matrix and vector file
 */
void MatrixAssemble(const CSRMatrix * /*matrix file data*/, Mat * /*petsc matrix*/);
void VectorAssemble(const CSRVector * /*vector file data*/, Vec * /*petsc vector*/);

/*
 * real/imaginary part of matrix
 */
void RealMatrixAssemble(const Mat * /*original petsc matrix*/, Mat * /*real part petsc matrix*/);
void ImaginaryMatrixAssemble(const Mat * /*original petsc matrix*/, Mat * /*imaginary part petsc matrix*/);

/*
 * real/imaginary part of vector
 */
void RealVectorAssemble(const Vec * /*original petsc vector*/, Vec * /*real part petsc vector*/);
void ImaginaryVectorAssemble(const Vec * /*original petsc vector*/, Vec * /*imaginary part pesc vector*/);

/*
* combined 2 vectors to 1 big vector
*/
void CombineTwoVectors(const Vec * /*petsc vector 1*/, const Vec * /*petsc vector 2*/,
                            Vec * /*combined vec1 and vec2*/);

#ifdef NEW_METHOD
/*
 * block matrix assemble
 */
void BlockPetscMatrixAssemble(const Mat * /*block(0, 0)*/, const Mat * /*block(0, 1)*/,
                              const Mat * /*block(1, 0)*/, const Mat * /*block(1, 1)*/,
                              Mat * /*block matrix*/);

/*
 * block vector assemble
 */
void BlockPetscVectorAssemble(const Vec * /*block(0, 0)*/, const Vec * /*block(0, 1)*/,
                              Vec * /*block vector*/);
#endif
#endif