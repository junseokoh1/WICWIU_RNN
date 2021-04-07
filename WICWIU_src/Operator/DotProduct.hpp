#ifndef DOTPRODUCT_H_
#define DOTPRODUCT_H_    value

#include "../Operator.hpp"
#include <cstdio>

template<typename DTYPE> class DotProduct : public Operator<DTYPE>{

public:
    /*!
    @brief MatMul의 생성자.
    @details 파라미터로 받은 pWeight와 pInput으로 Alloc한다.
    @param pWeight MatMul할 weight.
    @param pInput Matmul할 input Operator.
    @param pName 사용자가 부여한 Operator이름.
    @ref int Alloc(Operator<DTYPE> *pWeight, Operator<DTYPE> *pInput)
    */
    DotProduct(Operator<DTYPE> *pWeight, Operator<DTYPE> *pInput, std::string pName, int pLoadflag = TRUE) : Operator<DTYPE>(pWeight, pInput, pName, pLoadflag) {
        #ifdef __DEBUG__
        std::cout << "MatMul::MatMul(Operator<DTYPE> *, Operator<DTYPE> *, std::string)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pWeight, pInput);
    }

    /*!
    @brief MatMul의 소멸자
    @details Delete매소드를 사용해 GPU에 할당했던 값들을 해제한다.
    @ref void Delete()
    */
    virtual ~DotProduct() {
        #ifdef __DEBUG__
        std::cout << "Convolution2D::~Convolution2D()" << '\n';
        #endif  // __DEBUG__
        Delete();
    }


    int Alloc(Operator<DTYPE> *pWeight, Operator<DTYPE> *pInput) {
        #ifdef __DEBUG__
        std::cout << "MatMul::Alloc(Operator<DTYPE> *, Operator<DTYPE> *)" << '\n';
        #endif  // __DEBUG__

        int timesize    = pInput->GetResult()->GetTimeSize();
        int batchsize   = pInput->GetResult()->GetBatchSize();
        int channelsize = pInput->GetResult()->GetChannelSize();
        int rowsize     = pInput->GetResult()->GetRowSize();
        int colsize     = pWeight->GetResult()->GetRowSize();

        this->SetResult(new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, colsize));

        this->SetDelta(new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, colsize));

        return TRUE;
    }


    void Delete() {
    }


    int ForwardPropagate(int pTime = 0) {

        // std::cout<<"DotProduct Forward"<<'\n';

        Tensor<DTYPE> *weight = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *input  = this->GetInput()[1]->GetResult();
        Tensor<DTYPE> *result = this->GetResult();

        // std::cout<<"weight"<<'\n';
        // std::cout<<weight;
        //
        // std::cout<<"input"<<'\n';
        // std::cout<<input;

        int timesize    = result->GetTimeSize();
        int batchsize   = result->GetBatchSize();
        int channelsize = result->GetChannelSize();
        int rowsize     = result->GetRowSize();
        int colsize     = result->GetColSize();

        int hiddensize = input->GetColSize();             //이거 때문에 이제 forward는 정상적으로 진행됨!

        int weight_index = 0;
        int input_index  = 0;
        int result_index = 0;

        Shape *weightTenShape = weight->GetShape();
        Shape *inputTenShape  = input->GetShape();
        Shape *resultTenShape = result->GetShape();

        int ti = pTime;

        for (int ba = 0; ba < batchsize; ba++) {
            for (int ch = 0; ch < channelsize; ch++) {
                for (int ro = 0; ro < rowsize; ro++) {
                    for (int co = 0; co < colsize; co++) {
                        for (int hid = 0; hid < hiddensize; hid++) {
                            (*result)[Index5D(resultTenShape, ti, ba, ch, ro, co)]
                                += (*weight)[Index5D(weightTenShape, ti, ba, ch, co, hid)]
                                   * (*input)[Index5D(inputTenShape, ti, ba, ch, ro, hid)];
                            //std::cout<<(*result)[Index5D(resultTenShape, ti, ba, ch, ro, co)]<<" = "<<(*weight)[Index5D(weightTenShape, 0, 0, 0, co, hid)]<<" * "<<(*input)[Index5D(inputTenShape, ti, ba, ch, ro, hid)]<<'\n';
                        }
                    }
                }
            }
        }


        return TRUE;
    }


    int BackPropagate(int pTime = 0) {

        //std::cout<<"DotProduct의 Backpropagate"<<'\n';

        Tensor<DTYPE> *weight = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *input  = this->GetInput()[1]->GetResult();

        Tensor<DTYPE> *this_delta      = this->GetDelta();
        Tensor<DTYPE> *weight_gradient = this->GetInput()[0]->GetGradient();
        Tensor<DTYPE> *input_delta     = this->GetInput()[1]->GetDelta();

        int timesize    = this_delta->GetTimeSize();
        int batchsize   = this_delta->GetBatchSize();
        int channelsize = this_delta->GetChannelSize();
        int rowsize     = this_delta->GetRowSize();
        int colsize     = this_delta->GetColSize();
        int hiddensize  = input_delta->GetColSize();        //여기도 hiddensize존재!....

        Shape *weightTenShape = weight->GetShape();
        Shape *inputTenShape  = input->GetShape();
        Shape *resultTenShape = this_delta->GetShape();

        int weight_index = 0;
        int input_index  = 0;
        int result_index = 0;

        int ti = pTime;

       // std::cout<<resultTenShape<<'\n';

        for (int ba = 0; ba < batchsize; ba++) {
            for (int ch = 0; ch < channelsize; ch++) {
                for (int ro = 0; ro < rowsize; ro++) {
                    for (int co = 0; co < colsize; co++) {

                        //std::cout<<'\n';

                        for (int hid = 0; hid < hiddensize; hid++) {
                            weight_index = Index5D(weightTenShape, ti, ba, ch, co, hid);
                            input_index  = Index5D(inputTenShape, ti, ba, ch, ro, hid);
                            result_index = Index5D(resultTenShape, ti, ba, ch, ro, co);

                            (*input_delta)[input_index]      += (*weight)[weight_index] * (*this_delta)[result_index];

                            //std::cout<<ti<<" "<<ba<<" "<<ch<<" "<<ro<<" "<<hid<<'\n';
                            //std::cout<<(*input_delta)[input_index]<<"  +=  "<<(*weight)[weight_index]<<"  *  "<<(*this_delta)[result_index]<<'\n';

                            (*weight_gradient)[weight_index] += (*input)[input_index] * (*this_delta)[result_index];

                            //std::cout<<ti<<" "<<ba<<" "<<ch<<" "<<co<<" "<<hid<<'\n';
                            //std::cout<<(*weight_gradient)[weight_index]<<" += "<<(*input)[input_index]<<" * "<<(*this_delta)[result_index]<<'\n';


                        }
                    }
                }
            }
        }



        return TRUE;
    }

};


#endif  // MATMUL_H_
