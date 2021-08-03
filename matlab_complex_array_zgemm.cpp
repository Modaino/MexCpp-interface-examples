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
            int64_t inArray_A_size = 1;
            int64_t inArray_B_size = 1;
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
            auto ptr_A = dummyA.release(); //try with unique_ptr.get()
            auto ptr_B = dummyB.release();
            auto ptr_C = dummyC.get();
            const double * cPtr_A = (const double *) ptr_A;
            const double * cPtr_B = (const double *) ptr_B;
            double* cPtr_C = (double *) ptr_C;

            //Preparing constant arguments
            const int64_t m = (int64_t) dims_A[0];
            const int64_t n = (int64_t) dims_B[1];
            const int64_t k = (int64_t) dims_A[1];
            const int64_t lda = (int64_t) dims_A[0];
            const int64_t ldb = (int64_t) dims_B[0];
            const int64_t ldc = (int64_t) dims_A[0];



            //Calculation C := alpha* A * B + beta * C
            auto alpha = std::complex<double>(1,0);
            auto beta = std::complex<double>(0,0);
            const char* trans_A = "N";
            const char* trans_B = "N";

            // double* tmp = cPtr_C;

            // for (int64_t i = 0; i < inArray_A_size; i++)
            // {
            //     *tmp = (double) i;
            //     stream << *tmp << std::endl;
            //     tmp++;
            // }
            matlab_print(stream);

            //Calling blas function (currently does not compile)
            blas::zgemm(trans_A, trans_B,
                        &m, &n, &k, (const double *) &alpha,
                        cPtr_A, &lda,
                        cPtr_B, &ldb,
                        (const double*) &beta,
                        cPtr_C, &ldc);

            matlab_print(stream);

            outputs[0] = factory.createArrayFromBuffer(dims_A, std::move(dummyC));
        }
        else {
            stream << "Input is not of type COMPLEX_DOUBLE" << std::endl;
            matlab_print(stream);
        }
    }
};  
