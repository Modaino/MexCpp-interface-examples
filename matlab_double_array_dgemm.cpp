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
                stream << "Dimensions: " << element << std::endl; 
                inArray_size *= element;
            }
            stream << "Array size: " << inArray_size << std::endl; 

            //Preparing multiplication
            // double A[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
            // double B[9] = {1, 0, 0, 0, 2, 0, 0, 0, 3};
            // double C[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
            double* A = new double[9];
            double* B = new double[9];
            double* C = new double[9];

            stream << "The A matrix:" << std::endl;
            for (size_t i = 0; i < 3; i++) {
                for (size_t j = 0; j < 3; j++) { stream << A[3*i + j]; }
                stream << std::endl;
            }

            stream << "The B matrix:" << std::endl;
            for (size_t i = 0; i < 3; i++) {
                for (size_t j = 0; j < 3; j++) { stream << B[3*i + j]; }
                stream << std::endl;
            }
            
            mexprintf(stream);

            //Preparing constant arguments
            const size_t m = (size_t) 3;
            const size_t n = (size_t) 3;
            const size_t k = (size_t) 3;
            const size_t lda = (size_t) 3;
            const size_t ldb = (size_t) 3;
            const size_t ldc = (size_t) 3;

            const ptrdiff_t M = (const ptrdiff_t) &m;
            const ptrdiff_t N = (const ptrdiff_t) &n;
            const ptrdiff_t K = (const ptrdiff_t) &k;
            const ptrdiff_t LDA = (const ptrdiff_t) &lda;
            const ptrdiff_t LDB = (const ptrdiff_t) &ldb;
            const ptrdiff_t LDC = (const ptrdiff_t) &ldc;

            double alpha_ = 1.0;
            double beta_ = 1.0;
            //Shady casting (seems to be required)
            const double* alpha = &alpha_;
            const double* beta = &beta_;

            const char* trans_A = "n";
            const char* trans_B = "n";

            //Calculation C := alpha* A * B + beta * C

            //Calling blas function (does compile, but seems to fail)
            blas::dgemm(trans_A, trans_B,
                        &M, &N, &K, alpha,
                        (const double*) A, &LDA,
                        (const double*) B, &LDB,
                        beta,
                        (double *) C, &LDC);
                        
            //Preparing output
            matlab::data::buffer_ptr_t<double> dummy = inArray.release();
            outputs[0] = factory.createArrayFromBuffer(dims, std::move(dummy));

            //Writing multiplication result
            stream << "The C matrix:" << std::endl;
            for (size_t i = 0; i < 3; i++) {
                for (size_t j = 0; j < 3; j++) { stream << C[3*i + j]; }
                stream << std::endl;
            }
            mexprintf(stream);
            delete [] A;
            delete [] B;
            delete [] C;
        }
        else {
            stream << "I don't know what this is\n";
            mexprintf(stream);
        }
        return;
    }
};  
