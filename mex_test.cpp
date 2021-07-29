#include "mex.hpp"
#include "mexAdapter.hpp"
#include "MatlabDataArray.hpp"

#include "matarray_va.hpp"

#include <string>
//#include <complex>

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
        /* Structure: list info about all fields */
        if ( myType ==  matlab::data::ArrayType::STRUCT) {
            stream << "Input is a struct..." << std::endl << "complex array expeted" << std::endl;
            mexprintf(stream);

        }   

        /* Cell: list data info about all cells */
        else if ( myType ==  matlab::data::ArrayType::CELL ) {
            stream << "Input is a cell..." << std::endl;
            
            const CellArray myConstCell = inputs[0];
            constMatMatrixCells<TypedArray<double>> myMatMat(myConstCell);
            for (size_t i = 0; i < myMatMat.get_cols(); i++)
            {
                const TypedArray<double> elem_i = myMatMat(i);
                stream << "The 3,3 element of the "<< i << "th array is " << elem_i[2][2] << std::endl;
                
            }
            stream << "complex array expeted" << std::endl;
            mexprintf(stream);
        }   
        
        /* Other type: display size of data */
        else if ( myType == matlab::data::ArrayType::COMPLEX_DOUBLE ) {
            //source: https://www.mathworks.com/help/matlab/matlab_external/avoid-copies-of-large-arrays.html
            //const TypedArray<std::complex<double>> inArray = inputs[0];
            //const ensures that the variable is shared (no copy)

            TypedArray<std::complex<double>> inArray = std::move(inputs[0]);
            //auto outArray = factory.createArray<std::complex<double>>();

            size_t inArray_size = 1;
            
            auto dims = inArray.getDimensions();
            for (auto element : dims) {
                stream << "Dimensions: " << element << std::endl; 
                inArray_size *= element;
            }
            stream << "Array size: " << inArray_size << std::endl; 

            auto I = std::complex<double>(0,1);
            auto one = std::complex<double>(1,0);
            auto zero = std::complex<double>(0,0);

            std::complex<double> A[9] = {I, one, I};
            std::complex<double> B[9] = {one, I, one};
            std::complex<double> C[9] = {one, I, one};

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

            std::complex<double> alpha_ = one;
            std::complex<double> beta_ = zero;
            std::complex<double>* palpha = &alpha_;
            std::complex<double>* pbeta = &beta_;
            //Shady casting (seems to be required)
            const double* alpha = (const double*) palpha;
            const double* beta = (const double*) pbeta;

            const char* trans_A = "n";
            const char* trans_B = "n";

            //Calculation C := alpha* A * B + beta * C

            
            //Prepare ppointers
            const std::complex<double>* pA = A;
            const std::complex<double>* pB = B;
            std::complex<double>* pC = C;

            //Calling blas function (does compile, but seems to fail)
            blas::zgemm(trans_A, trans_B,
                        &M, &N, &K, alpha,
                        (const double*) pA, &LDA,
                        (const double*) pB, &LDB,
                        beta,
                        (double *) pC, &LDC);
                        
            //Preparing output
            matlab::data::buffer_ptr_t<std::complex<double>> dummy = inArray.release();
            outputs[0] = factory.createArrayFromBuffer(dims, std::move(dummy));
            stream << "The C matrix:" << std::endl;
            for (size_t i = 0; i < 3; i++) {
                for (size_t j = 0; j < 3; j++) { stream << C[3*i + j]; }
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
