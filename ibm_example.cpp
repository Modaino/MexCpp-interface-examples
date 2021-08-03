#include <stdio.h>
#include <time.h>
#include <stdlib.h>

#include "mex.hpp"
#include "mexAdapter.hpp"
#include "MatlabDataArray.hpp"

#include <string>

using namespace matlab::data;
using matlab::mex::ArgumentList;

namespace blas {
    #include "blas.h"
}

class MexFunction : public matlab::mex::Function {
    public:
    // Pointer to MATLAB engine to call fprintf
    std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr = getEngine();

    std::ostringstream stream;
    // Factory to create MATLAB data arrays
    ArrayFactory factory;

    void mexprintf(std::ostringstream& stream_){
        // Pass stream content to MATLAB fprintf function
        matlabPtr->feval(u"fprintf", 0,
            std::vector<Array>({ factory.createScalar(stream.str()) }));
        // Clear stream buffer
        stream.str("");
    }

	void init(double* matrix, int row, int column)
	{
		for (int j = 0; j < column; j++){
			for (int i = 0; i < row; i++){
				matrix[j*row + i] = ((double)rand())/RAND_MAX;
			}
		}
	}

    void operator()(ArgumentList outputs, ArgumentList inputs) {
        size_t nrhs = inputs.size();
        if (outputs.size() != 1) {
            matlabPtr->feval(u"error", 0, std::vector<matlab::data::Array>({ factory.createScalar("One output is required") }));
        }
        if (inputs.size() != 1) {
            matlabPtr->feval(u"error", 0, std::vector<matlab::data::Array>({ factory.createScalar("One input is required") }));
        }

        auto myType = inputs[0].getType();
        if ( myType == matlab::data::ArrayType::DOUBLE ) {
            
            //Checking input
            TypedArray<double> inArray = std::move(inputs[0]);
            
            size_t inArray_size = 1;
            auto dims = inArray.getDimensions();
            for (auto element : dims) {
                //stream << "Dimensions: " << element << std::endl; 
                inArray_size *= element;
            }
            //stream << "Array size: " << inArray_size << std::endl; 

			//stream << "The size of size_t: " << sizeof(inArray_size) << std::endl;

			int64_t rowsA, colsB, common;
			int64_t i,j,k;
			rowsA = 2; colsB = 4; common = 6;

			double A[rowsA * common]; double B[common * colsB];
			double C[rowsA * colsB]; double D[rowsA * colsB];

			char transA = 'N', transB = 'N';
			double one = 1.0, zero = 0.0;

			init(A, rowsA, common); init(B, common, colsB);

			mexprintf(stream);

			blas::dgemm(&transA, &transB, &rowsA, &colsB, &common, &one, A, 
			&rowsA, B, &common, &zero, C, &rowsA);

			for(i=0;i<colsB;i++){
				for(j=0;j<rowsA;j++){
					D[i*rowsA+j]=0;
					for(k=0;k<common;k++){
						D[i*rowsA+j]+=A[k*rowsA+j]*B[k+common*i];
					}
				}
			}
                        
            //Preparing output
            matlab::data::buffer_ptr_t<double> dummy = inArray.release();
            outputs[0] = factory.createArrayFromBuffer(dims, std::move(dummy));

            //Writing multiplication result
            stream << "The A matrix:" << std::endl;
            for (i = 0; i < rowsA; i++) {
                for (j = 0; j < common; j++) { stream << A[rowsA*i + j]; }
                stream << std::endl;
            }
			stream << "The B matrix:" << std::endl;
            for (i = 0; i < common; i++) {
                for (j = 0; j < colsB; j++) { stream << B[common*i + j]; }
                stream << std::endl;
            }
			stream << "The C matrix:" << std::endl;
            for (i = 0; i < rowsA; i++) {
                for (j = 0; j < colsB; j++) { stream << C[rowsA*i + j]; }
                stream << std::endl;
            }
			stream << "The D matrix:" << std::endl;
            for (i = 0; i < rowsA; i++) {
                for (j = 0; j < colsB; j++) { stream << D[rowsA*i + j]; }
                stream << std::endl;
            }
			mexprintf(stream);
        }
        else {
            stream << "I don't know what this is\n";
            mexprintf(stream);
        }
        return;
    }
};