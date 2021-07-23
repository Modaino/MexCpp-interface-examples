#include "mex.hpp"
#include "mexAdapter.hpp"
#include "MatlabDataArray.hpp"

#include <string>
namespace blas {
    #include "blas.h"
}

using namespace matlab::data;
using matlab::mex::ArgumentList;

class MexFunction : public matlab::mex::Function {
    public:
    // Pointer to MATLAB engine to call fprintf
    std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr = getEngine();

    std::ostringstream stream;
    // Factory to create MATLAB data arrays
    ArrayFactory factory;

    void matlab_print(std::ostringstream& stream_){
        // Pass stream content to MATLAB fprintf function
        matlabPtr->feval(u"fprintf", 0,
            std::vector<Array>({ factory.createScalar(stream.str()) }));
        // Clear stream buffer
        stream.str("");
    }

    void operator()(ArgumentList outputs, ArgumentList inputs) {
        //Quick chck on input/output arguments (can cause segmentation fault otherwise)
        if (outputs.size() != 1) {
            matlabPtr->feval(u"error", 0, std::vector<matlab::data::Array>({ factory.createScalar("One output is required") }));
        }
        if (inputs.size() != 2) {
            matlabPtr->feval(u"error", 0, std::vector<matlab::data::Array>({ factory.createScalar("One input is required") }));
        }

        if ( inputs[0].getType() == matlab::data::ArrayType::COMPLEX_DOUBLE && inputs[1].getType() == matlab::data::ArrayType::COMPLEX_DOUBLE) {
            //Handling input variables without copy.
            TypedArray<std::complex<double>> inArray_A = std::move(inputs[0]);
            TypedArray<std::complex<double>> inArray_B = std::move(inputs[1]);

            //Checking dimensions
            size_t inArray_A_size = 1;
            size_t inArray_B_size = 1;
            auto dims_A = inArray_A.getDimensions();
            auto dims_B = inArray_B.getDimensions();
            for (auto element : dims_A) {
                stream << "Dimensions of A: " << element << std::endl; 
                inArray_A_size *= element;
            }
            stream << "Array A size: " << inArray_A_size << std::endl; 
            for (auto element : dims_B) {
                stream << "Dimensions of B: " << element << std::endl; 
                inArray_B_size *= element;
            }
            stream << "Array B size: " << inArray_B_size << std::endl; 
            if (inArray_A_size != inArray_B_size){
                matlabPtr->feval(u"error", 0, std::vector<matlab::data::Array>({ factory.createScalar("The sizes do not match!") }));
            }
            
            //Retrieving (unique) pointers to input variables. Ownership of underlying data is no longer with the arrays.
            matlab::data::buffer_ptr_t<std::complex<double>> dummyA = inArray_A.release();
            matlab::data::buffer_ptr_t<std::complex<double>> dummyB = inArray_B.release();

            //Preparing output variable
            auto dummyC = factory.createBuffer<std::complex<double>>(inArray_A_size);

            //Preparing (and casting) pointers to pass to blass call 
            auto ptr_A = dummyA.get();
            auto ptr_B = dummyB.get();
            auto ptr_C = dummyC.get();
            double* cPtr_A = (double *) ptr_A;
            double* cPtr_B = (double *) ptr_B;
            double* cPtr_C = (double *) ptr_C;

            //Preparing constant arguments
            size_t m = (size_t) dims_A[0];
            size_t n = (size_t) dims_B[1];
            size_t k = (size_t) dims_A[1];
            size_t lda = (size_t) dims_A[0];
            size_t ldb = (size_t) dims_B[0];
            size_t ldc = (size_t) dims_A[0];

            ptrdiff_t M = (ptrdiff_t) &m;
            ptrdiff_t N = (ptrdiff_t) &n;
            ptrdiff_t K = (ptrdiff_t) &k;
            ptrdiff_t LDA = (ptrdiff_t) &lda;
            ptrdiff_t LDB = (ptrdiff_t) &ldb;
            ptrdiff_t LDC = (ptrdiff_t) &ldc;


            //Calculation C := alpha* A * B + beta * C
            auto alpha = std::complex<double>(1,0);
            auto beta = std::complex<double>(1,0);
            char * trans_A; trans_A[0] = 'n';
            char * trans_B; trans_B[0] = 'n';
            
            //Calling blas function (currently does not compile)
            blas::zgemm(trans_A, trans_B,
                        &M, &N, &K, (double *) &alpha,
                        cPtr_A, &LDA,
                        cPtr_B, &LDB,
                        (double*) &beta,
                        cPtr_C, &LDC);


            outputs[0] = factory.createArrayFromBuffer(dims_A, std::move(dummyC));
            matlab_print(stream);
        }
        else {
            stream << "Input is not of type COMPLEX_DOUBLE" << std::endl;
            matlab_print(stream);
        }
    }
};  
