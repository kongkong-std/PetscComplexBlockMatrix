#ifndef MAIN_H_
#define MAIN_H_

#include <petscksp.h>
#include <mpi.h>
#include "file_process.h"

// function prototype
void MatrixAssemble(const CSRMatrix * /*matrix file data*/, Mat * /*petsc matrix*/);
void VectorAssemble(const CSRVector * /*vector file data*/, Vec * /*petsc vector*/);
void RealMatrixAssemble(const Mat * /*original petsc matrix*/, Mat * /*real part petsc matrix*/);
void ImaginaryMatrixAssemble(const Mat * /*original petsc matrix*/, Mat * /*imaginary part petsc matrix*/);
void RealVectorAssemble(const Vec * /*original petsc vector*/, Vec * /*real part petsc vector*/);
void ImaginaryVectorAssemble(const Vec * /*original petsc vector*/, Vec * /*imaginary part pesc vector*/);

#endif