// Installation: Install openblas-dev in linux, Accelerate framework in Mac to get BLAS and LAPACK functions
// Compilation: g++ -O2 cholesky.cpp -o cholesky -fopenmp -lblas -llapack
// Execution: ./cholesky
// For more info regarding BLAS/LAPACK functions, c.f. Intel MKL documentation
// Pour plus d'information concernant les fonctions dans BLAS/LAPACK, c.f. documentation d'Intel MKL
//   https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-fortran/2024-2
// Information on blocked Cholesky algorithm
// Information sur l'algorithme de Cholesky par blocs
//   https://www.netlib.org/utk/papers/factor/node9.html
// Complete list of BLAS routines
// Liste complete des routines BLAS
//   https://www.netlib.org/blas/blasqr.pdf

#include <iostream>
#include <vector>

// 2-norm of a vector (BLAS1)
// 2-norme pour un vecteur (BLAS1)
extern "C" double dnrm2_(
  int *n,
  double *x,
  int *incx);

// Compute y = y + a x, where x and y are vectors and a is a scalar (BLAS1)
// Calculer y = y + a x, x et y etant des vecteurs et a etant un scalaire (BLAS1)
extern "C" void daxpy_(
  int *n,
  double *a,
  double *x,
  int *incx,
  double *y,
  int *incy);

// Matrix-vector multiplication where alpha and beta are scalars (BLAS2)
// Produit matrice-vecteur, alpha et beta etant des scalaires (BLAS2)
// y = alpha A x + beta y
extern "C" void dgemv_(
  char *trans,
  int *m,
  int *n,
  double *alpha,
  double *a,
  int *lda,
  double *x,
  int *incx,
  double *beta,
  double *y,
  int *incy);

// Lower/upper triangular system solve for a single column vector. x is modified in-place (overwritten by x_new)
// Resoudre un systeme triangulaire superieure/inferieure pour un vecteur de colonne. x est modifie en place (surecrit
// par x_new)
// L x_new = x, U x_new = x (BLAS2)
extern "C" void dtrsv_(
  char *uplo,
  char *trans,
  char *diag,
  int *n,
  double *A,
  int *lda,
  double *x,
  int *incx);

// Matrix-matrix multiplication, alpha and beta are scalars and op(.) is an optional matrix transposition operator.
// Produit matrice-vecteur, alpha et beta sont des scalaires et op(.) est une transposition matricielle optionnelle.
// C = alpha opA(A) op(B) + beta op(C) (BLAS3)
extern "C" void dgemm_(
  char *transA,
  char *transB,
  int *m,
  int *n,
  int *k,
  double *alpha,
  double *A,
  int *lda,
  double *B,
  int *ldb,
  double *beta,
  double *C,
  int *ldc);

// Symmetric matrix-matrix multiplication. A itself does not have to be symmetric
// Produit matrice-matrice symmetrique. A elle-meme n'a pas a etre symmetrique
// C = alpha A A^T + beta C, C = alpha A^T A + beta C
extern "C" void dsyrk_(
  char *uplo,
  char *trans,
  int *n,
  int *k,
  double *alpha,
  double *A,
  int *lda,
  double *beta,
  double *C,
  int *ldc);

// Cholesky factorization L L^T = A or U UˆT = A (LAPACK). A is overwritten by L or L^T
// Once computed, you can solve A x = b as follows:
// Factorisation Cholesky L L^T = A or U UˆT = A (LAPACK). A est surecrit par L ou L^T
// Une fois calculee, on peut resoudre A x = B comme suit:
// A x = b 
// L L^T x = b         -> dpotrf
// L L^T x = L y = b   -> dtrsv
// L^T x = y           -> dtrsv
extern "C" void dpotrf_(
  char *uplo,
  int *n,
  double *A,
  int *lda,
  int *info);

void printMatrix(
  const std::vector<double>& matrix,
  int m,
  int n)
{
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) { printf("%5.1lf", matrix[i + j * n]); }
    printf("\n");
  }
  printf("\n");
}

int main()
{
  // Dimension of matrices / Dimensions des matrices
  int N = 8; 
  // Block size for the task-parallel blocked potrf code / Taille de bloc pour potrf parallele par bloc a base de taches
  int BS = 4;  
  // Matrices
  std::vector<double> L(N * N), A(N * N), B(N * N);
  // Vectors
  std::vector<double> x(N), b(N), b2(N);

  // Generate a lower-triangular N x N matrix L with random values between 0.0 and 1.0
  for (int j = 0; j < N; ++j) {
    for (int i = 0; i < N; ++i) {
      if (i >= j) {
        L[j * N + i] = static_cast<double>(std::rand()) / RAND_MAX; // Column-major
      }
    }
  }
  for (int i = 0; i < N; ++i) {
    b[i] = static_cast<double>(std::rand()) / RAND_MAX;
  }
  std::cout << "Matrix L:" << std::endl;
  printMatrix(L, N, N);
  std::cout << "Vector b:" << std::endl;
  printMatrix(b, N, 1);

  // Generate a symmetric positive definite matrix A = L * L^T using the dsyrk function (BLAS3)
  char trans = 'N';
  char transT = 'T';
  double alpha = 1.0;
  double beta = 0.0;
  dgemm_(&trans, &transT, &N, &N, &N, &alpha, &L[0], &N, &L[0], &N, &beta, &A[0], &N);
  std::cout << "Matrix A (L * L^T):" << std::endl;
  printMatrix(A, N, N);

  // Perform a Cholesky factorization on the matrix A, A = L LˆT using the potrf function (LAPACK)
  char uplo = 'L';
  int info;
  B = A;
  dpotrf_(&uplo, &N, &A[0], &N, &info);
  if (info != 0) {
    std::cerr << "dpotrf failed with info = " << info << std::endl;
    return 1;
  }
  std::cout << "Matrix A (Cholesky factorization):" << std::endl;
  printMatrix(A, N, N);

  // Solve the linear system A x = L L^T x = b by first solving L y = b, then solving LˆT x = y, with two successive calls to dtrsv
  std::copy(b.begin(), b.end(), b2.begin());
  char diag = 'N';
  int incx = 1;
  dtrsv_(&uplo, &trans, &diag, &N, &A[0], &N, &b2[0], &incx);
  std::cout << "Solution vector L y = b:" << std::endl;
  printMatrix(b2, N, 1);

  dtrsv_(&uplo, &transT, &diag, &N, &A[0], &N, &b2[0], &incx);
  std::copy(b2.begin(), b2.end(), x.begin());

  // Verify the solution x by computing b2 = A x using dgemv, then compare it to the initial right hand side vector by
  // computing (b - b2) using daxpy, and computing the norm of this vector~(which is the error) by dnrm2
  alpha = 1.0;
  beta = 0.0;
  dgemv_(&trans, &N, &N, &alpha, &B[0], &N, &x[0], &incx, &beta, &b2[0], &incx);
  std::cout << "Vector b2 (verification):" << std::endl;
  printMatrix(b2, N, 1);

  alpha = -1.0;
  daxpy_(&N, &alpha, &b[0], &incx, &b2[0], &incx);
  std::cout << "Vector b2 - b:" << std::endl;
  printMatrix(b2, N, 1);

  double error = dnrm2_(&N, &b2[0], &incx);
  std::cout << "error norm: " << error << "\n" << std::endl;

  return 0;
}
