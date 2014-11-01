#include <algorithm>
#include <fstream>
#include <iostream>
#include <cmath>
#include <map>
#include <stdio.h>
#include <stdlib.h>
#include <string>

#include <cusp/io/matrix_market.h>
#include <cusp/csr_matrix.h>
#include <cusp/multiply.h>
#include <cusp/blas.h>
#include <cusp/detail/format_utils.h>
#include <cusp/detail/host/convert.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cusparse/common.h>
#include <cusparse/timer.h>
#include "cusparse.h"

using std::endl;
using std::cerr;
using std::cout;

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
#ifdef WIN32
#   define ISNAN(A)  (_isnan(A))
#else
#   define ISNAN(A)  (isnan(A))
#endif

enum TestColor {COLOR_NO = 0,
                COLOR_RED,
                COLOR_GREEN} ;

class OutputItem
{
public:
	OutputItem(std::ostream &o): m_o(o), m_additional_item_count(19) {}

	int           m_additional_item_count;

	template <typename T>
	void operator() (T item, TestColor c = COLOR_NO) {
		m_o << "<td style=\"border-style: inset;\">\n";
		switch (c)
		{
			case COLOR_RED:
				m_o << "<p> <FONT COLOR=\"Red\">" << item << " </FONT> </p>\n";
				break;

			case COLOR_GREEN:
				m_o << "<p> <FONT COLOR=\"Green\">" << item << " </FONT> </p>\n";
				break;

			default:
				m_o << "<p> " << item << " </p>\n";
				break;
		}
		m_o << "</td>\n";
	}
private:
	std::ostream &m_o;
};

int main(int argc, char **argv)
{
	if (argc < 2) {
		cerr << "Usage: ./example MATRIX_MARKET_FILE_NAME" << endl;
		return 1;
	}

	cusp::csr_matrix<int, double, cusp::device_memory> Ad_cusp;
	cusp::io::read_matrix_market_file(Ad_cusp, argv[1]);

	cusparse::CuSparseCsrMatrixD A(Ad_cusp.row_offsets, Ad_cusp.column_indices, Ad_cusp.values);
	cusparse::CuSparseCsrMatrixD Abak(Ad_cusp.row_offsets, Ad_cusp.column_indices, Ad_cusp.values);

	thrust::device_vector<double> x(A.m_n, 1.0);
	thrust::device_vector<double> y;
	thrust::device_vector<double> x_new(A.m_n);

	// Name of matrix
	OutputItem outputItem(cout);

	cout << "<tr valign=top>" << endl;

	// Name of matrix
	{
		std::string fileMat = argv[1];
		int i;
		for (i = fileMat.size()-1; i>=0 && fileMat[i] != '/' && fileMat[i] != '\\'; i--);
		i++;
		fileMat = fileMat.substr(i);

		size_t j = fileMat.rfind(".mtx");
		if (j != std::string::npos)
			outputItem( fileMat.substr(0, j));
		else
			outputItem( fileMat);
	}

	// Dimension
	outputItem( A.m_n);
	//aNNZ 
	outputItem( A.m_nnz);

	cusparse::GPUTimer local_timer;
	local_timer.Start();
	cusolverStatus_t status = A.QRSolve(x, y);
	local_timer.Stop();

	int code = status;

	// Error code
	if (status != CUSOLVER_STATUS_SUCCESS) {
		outputItem(code, COLOR_RED);
		outputItem("");
		outputItem("");
		cout << "</tr>" << endl;
		return 1;
	} else
		outputItem(int(0));

	Abak.spmv(y, x_new);

	// The relative infinity norm of solution
	double nrm_target = cusp::blas::nrmmax(x);
	cusp::blas::axpy(x, x_new, (double)(-1));
	double rel_err = fabs(cusp::blas::nrmmax(x_new))/ nrm_target;
	if (isnan(cusp::blas::nrm1(x_new)))
		outputItem("NaN", COLOR_RED);
	else if (rel_err >= 1)
		outputItem(rel_err, COLOR_RED);
	else
		outputItem(rel_err);

	outputItem( local_timer.getElapsed());
	cout << "</tr>" << endl;
 

	return 0;
}
