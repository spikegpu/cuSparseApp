#ifndef COMMON_H
#define COMMON_H



#define ALWAYS_ASSERT

#ifdef WIN32
typedef long long int64_t;
#endif



// ----------------------------------------------------------------------------
// If ALWAYS_ASSERT is defined, we make sure that  assertions are triggered 
// even if NDEBUG is defined.
// ----------------------------------------------------------------------------
#ifdef ALWAYS_ASSERT
// If NDEBUG is actually defined, remember this so
// we can restore it.
#  ifdef NDEBUG
#    define NDEBUG_ACTIVE
#    undef NDEBUG
#  endif
// Include the assert.h header file here so that it can
// do its stuff while NDEBUG is guaranteed to be disabled.
#  include <assert.h>
// Restore NDEBUG mode if it was active.
#  ifdef NDEBUG_ACTIVE
#    define NDEBUG
#    undef NDEBUG_ACTIVE
#  endif
#else
// Include the assert.h header file using whatever the
// current definition of NDEBUG is.
#  include <assert.h>
#endif

# include <memory.h>
# include <thrust/device_vector.h>
# include <thrust/host_vector.h>
# include <thrust/device_ptr.h>
# include <thrust/system/cuda/execution_policy.h>
# include "cusparse.h"
# include "cusolver.h"

# define TEMP_TOL 1e-10


// ----------------------------------------------------------------------------


namespace cusparse {

const unsigned int BLOCK_SIZE = 512;

const unsigned int MAX_GRID_DIMENSION = 32768;

inline
void kernelConfigAdjust(int &numThreads, int &numBlockX, const int numThreadsMax) {
	if (numThreads > numThreadsMax) {
		numBlockX = (numThreads + numThreadsMax - 1) / numThreadsMax;
		numThreads = numThreadsMax;
	}
}

inline
void kernelConfigAdjust(int &numThreads, int &numBlockX, int &numBlockY, const int numThreadsMax, const int numBlockXMax) {
	kernelConfigAdjust(numThreads, numBlockX, numThreadsMax);
	if (numBlockX > numBlockXMax) {
		numBlockY = (numBlockX + numBlockXMax - 1) / numBlockXMax;
		numBlockX = numBlockXMax;
	}
}

class CuSparseCsrMatrixD
{
public:
	CuSparseCsrMatrixD(int N, int nnz): m_n(N), m_nnz(nnz), m_l_analyzed(false), m_u_analyzed(false), m_tolerance(TEMP_TOL) {
		cudaMalloc(&m_row_offsets,   sizeof(int) * (N + 1));
		cudaMalloc(&m_column_indices, sizeof(int) * nnz);
		cudaMalloc(&m_values,         sizeof(double) * nnz);

		if (!m_handle_initialized) {
			cusparseCreate(&m_handle);
			m_handle_initialized = true;
		}

		if (!m_solver_handle_initialized) {
			cusolverCreate(&m_solver_handle);
			m_solver_handle_initialized = true;
		}

		cusparseCreateSolveAnalysisInfo(&m_infoL);
		cusparseCreateSolveAnalysisInfo(&m_infoU);
		cusparseCreateMatDescr(&m_descrL);
		cusparseCreateMatDescr(&m_descrU);
	}

	CuSparseCsrMatrixD(int N, int nnz, int *row_offsets, int *column_indices, double *values): m_n(N), m_nnz(nnz), m_l_analyzed(false), m_u_analyzed(false), m_tolerance(TEMP_TOL) {
		cudaMalloc(&m_row_offsets,   sizeof(int) * (N + 1));
		cudaMemcpy(m_row_offsets, row_offsets, sizeof(int) * (N + 1), cudaMemcpyDeviceToDevice);

		cudaMalloc(&m_column_indices, sizeof(int) * nnz);
		cudaMemcpy(m_column_indices, column_indices, sizeof(int) * nnz, cudaMemcpyDeviceToDevice);

		cudaMalloc(&m_values,         sizeof(double) * nnz);
		cudaMemcpy(m_values, values, sizeof(double) * nnz, cudaMemcpyDeviceToDevice);

		if (!m_handle_initialized) {
			cusparseCreate(&m_handle);
			m_handle_initialized = true;
		}

		if (!m_solver_handle_initialized) {
			cusolverCreate(&m_solver_handle);
			m_solver_handle_initialized = true;
		}

		cusparseCreateSolveAnalysisInfo(&m_infoL);
		cusparseCreateSolveAnalysisInfo(&m_infoU);
		cusparseCreateMatDescr(&m_descrL);
		cusparseCreateMatDescr(&m_descrU);
	}

	template<typename IntVector, typename DoubleVector>
	CuSparseCsrMatrixD(const IntVector    &row_offsets,
			           const IntVector    &column_indices,
					   const DoubleVector &values): m_l_analyzed(false), m_u_analyzed(false), m_tolerance(TEMP_TOL) {
		int N   = row_offsets.size() - 1;
		m_n     = N;
		int nnz = column_indices.size();
		m_nnz   = nnz;
		const int *p_row_offsets   = thrust::raw_pointer_cast(&row_offsets[0]);
		const int *p_column_indices = thrust::raw_pointer_cast(&column_indices[0]);
		const double *p_values         = thrust::raw_pointer_cast(&values[0]);

		if (!m_handle_initialized) {
			cusparseCreate(&m_handle);
			m_handle_initialized = true;
		}

		if (!m_solver_handle_initialized) {
			cusolverCreate(&m_solver_handle);
			m_solver_handle_initialized = true;
		}

		cudaMalloc(&m_row_offsets,   sizeof(int) * (N + 1));
		cudaMemcpy(m_row_offsets, p_row_offsets, sizeof(int) * (N + 1), cudaMemcpyDeviceToDevice);

		cudaMalloc(&m_column_indices, sizeof(int) * nnz);
		cudaMemcpy(m_column_indices, p_column_indices, sizeof(int) * nnz, cudaMemcpyDeviceToDevice);

		cudaMalloc(&m_values,         sizeof(double) * nnz);
		cudaMemcpy(m_values, p_values, sizeof(double) * nnz, cudaMemcpyDeviceToDevice);

		cusparseCreateSolveAnalysisInfo(&m_infoL);
		cusparseCreateSolveAnalysisInfo(&m_infoU);
		cusparseCreateMatDescr(&m_descrL);
		cusparseCreateMatDescr(&m_descrU);
	}

	virtual ~CuSparseCsrMatrixD() {
		if (m_row_offsets)    cudaFree(m_row_offsets);
		if (m_column_indices) cudaFree(m_column_indices);
		if (m_values)         cudaFree(m_values);

		cusparseDestroySolveAnalysisInfo(m_infoL);
		cusparseDestroySolveAnalysisInfo(m_infoU);
		cusparseDestroyMatDescr(m_descrL);
		cusparseDestroyMatDescr(m_descrU);
		m_infoL = 0;
		m_infoU = 0;
		m_descrL = 0;
		m_descrU = 0;
	}

	int    m_n;
	int    m_nnz;
	int    *m_row_offsets;
	int    *m_column_indices;
	double *m_values;

	template<typename DVector>
	cusparseStatus_t spmv(const DVector &x, DVector &y)
	{
		if (y.size() != x.size())
			y.resize(x.size());

		double one = 1.0, zero = 0.0;
		const double *p_x = thrust::raw_pointer_cast(&x[0]);
		double *p_y = thrust::raw_pointer_cast(&y[0]);

		cusparseSetMatType(m_descrL,CUSPARSE_MATRIX_TYPE_GENERAL);
		cusparseSetMatDiagType(m_descrL,CUSPARSE_DIAG_TYPE_NON_UNIT);
		cusparseSetMatIndexBase(m_descrL,CUSPARSE_INDEX_BASE_ZERO);
		cusparseSetMatFillMode(m_descrL, CUSPARSE_FILL_MODE_LOWER);

		return cusparseDcsrmv(m_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, m_n, m_n, m_nnz, &one, m_descrL, m_values, m_row_offsets, m_column_indices, p_x, &zero, p_y);
	}

	template<typename DVector>
	cusparseStatus_t forwardSolve(const DVector &x, DVector &y, bool unit = true)
	{
		if (y.size() != x.size())
			y.resize(x.size());

		double one = 1.0;
		const double *p_x = thrust::raw_pointer_cast(&x[0]);
		double *p_y = thrust::raw_pointer_cast(&y[0]);

		cusparseSetMatType(m_descrL,CUSPARSE_MATRIX_TYPE_GENERAL);
		if (unit)
			cusparseSetMatDiagType(m_descrL,CUSPARSE_DIAG_TYPE_UNIT);
		else
			cusparseSetMatDiagType(m_descrL,CUSPARSE_DIAG_TYPE_NON_UNIT);

		cusparseSetMatIndexBase(m_descrL,CUSPARSE_INDEX_BASE_ZERO);
		cusparseSetMatFillMode(m_descrL, CUSPARSE_FILL_MODE_LOWER);

		cusparseStatus_t status;

		if (!m_l_analyzed) {
			status = cusparseDcsrsv_analysis(m_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, m_n, m_nnz, m_descrL, m_values, m_row_offsets, m_column_indices, m_infoL);
			if (status != CUSPARSE_STATUS_SUCCESS)
				return status;
		}

		status = cusparseDcsrsv_solve(m_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, m_n, &one, m_descrL, m_values, m_row_offsets, m_column_indices, m_infoL, p_x, p_y);

		return status;
	}

	template<typename DVector>
	cusparseStatus_t backwardSolve(const DVector &x, DVector &y, bool unit = false)
	{
		if (y.size() != x.size())
			y.resize(x.size());

		double one = 1.0;
		const double *p_x = thrust::raw_pointer_cast(&x[0]);
		double *p_y = thrust::raw_pointer_cast(&y[0]);

		cusparseSetMatType(m_descrU,CUSPARSE_MATRIX_TYPE_GENERAL);
		if (unit)
			cusparseSetMatDiagType(m_descrU,CUSPARSE_DIAG_TYPE_UNIT);
		else
			cusparseSetMatDiagType(m_descrU,CUSPARSE_DIAG_TYPE_NON_UNIT);

		cusparseSetMatIndexBase(m_descrU,CUSPARSE_INDEX_BASE_ZERO);
		cusparseSetMatFillMode(m_descrU, CUSPARSE_FILL_MODE_UPPER);

		cusparseStatus_t status;

		if (!m_u_analyzed) {
			status = cusparseDcsrsv_analysis(m_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, m_n, m_nnz, m_descrU, m_values, m_row_offsets, m_column_indices, m_infoU);
			if (status != CUSPARSE_STATUS_SUCCESS)
				return status;
		}

		status = cusparseDcsrsv_solve(m_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, m_n, &one, m_descrU, m_values, m_row_offsets, m_column_indices, m_infoU, p_x, p_y);

		return status;
	}

	template<typename DVector>
	cusolverStatus_t QRSolve(const DVector &x, DVector &y)
	{
		if (y.size() != x.size())
			y.resize(x.size());

		const double *p_x = thrust::raw_pointer_cast(&x[0]);
		double *p_y = thrust::raw_pointer_cast(&y[0]);

		cusparseSetMatType(m_descrL,CUSPARSE_MATRIX_TYPE_GENERAL);
		cusparseSetMatDiagType(m_descrL,CUSPARSE_DIAG_TYPE_NON_UNIT);

		cusparseSetMatIndexBase(m_descrL,CUSPARSE_INDEX_BASE_ZERO);
		cusparseSetMatFillMode(m_descrL, CUSPARSE_FILL_MODE_LOWER);

		cusolverStatus_t status;

		int singularity;

		status = cusolverDcsrlsvqr(m_solver_handle, m_n, m_nnz, m_descrL, m_values, m_row_offsets, m_column_indices, p_x, m_tolerance, p_y, &singularity);

		return status;
	}

private:
	bool   m_l_analyzed;
	bool   m_u_analyzed;
	cusparseSolveAnalysisInfo_t m_infoL;
	cusparseSolveAnalysisInfo_t m_infoU;
	cusparseMatDescr_t          m_descrL;
	cusparseMatDescr_t          m_descrU;

	double m_tolerance;

	static cusparseHandle_t m_handle;
	static cusolverHandle_t m_solver_handle;
	static bool             m_handle_initialized;
	static bool             m_solver_handle_initialized;
};

cusparseHandle_t CuSparseCsrMatrixD::m_handle = 0;
bool             CuSparseCsrMatrixD::m_handle_initialized = false;
cusolverHandle_t CuSparseCsrMatrixD::m_solver_handle = 0;
bool             CuSparseCsrMatrixD::m_solver_handle_initialized = false;


} // namespace cusparse


#endif
