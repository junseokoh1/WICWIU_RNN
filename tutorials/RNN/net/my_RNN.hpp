#include <iostream>
#include <string>

#include "../../../WICWIU_src/NeuralNetwork.hpp"


class my_RNN : public NeuralNetwork<float>{
private:
public:
    my_RNN(Tensorholder<float> *x, Tensorholder<float> *label, int vocab_length) {
        SetInput(2, x, label);

        Operator<float> *out = NULL;

        //out = new ReShape<float>(x, 28, 28, "Flat2Image");

        //out = new CBOW<float>(x(입력 배열), 아웃풋크기, "CBOW");
        //out = new OnehotVector<float>(x(입력 배열), 아웃풋크기, "OnehotVector");

        // ======================= layer 1=======================
        out = new RecurrentLayer<float>(x, vocab_length, 32, vocab_length, TRUE, "Recur_1");
        //out = new DeepRecurrentLayer<float>(x, vocab_length, 128, vocab_length, TRUE, "Recur_1");

        //out = new LSTMLayer<float>(x, vocab_length, 32, vocab_length, TRUE, "Recur_1");
        //out = new LSTM2Layer<float>(x, vocab_length, 128, vocab_length, TRUE, "Recur_1");

        //out = new GRULayer<float>(x, vocab_length, 128, vocab_length, TRUE, "Recur_1");

        // // ======================= layer 2=======================
        // out = new Linear<float>(out, 5 * 5 * 20, 1024, TRUE, "Fully-Connected_1");

        AnalyzeGraph(out);

        // ======================= Select LossFunction Function ===================
        // SetLossFunction(new HingeLoss<float>(out, label, "HL"));
        // SetLossFunction(new MSE<float>(out, label, "MSE"));
        SetLossFunction(new SoftmaxCrossEntropy<float>(out, label, "SCE"));
        // SetLossFunction(new CrossEntropy<float>(out, label, "CE"));

        // ======================= Select Optimizer ===================
        // 1.0이 clipValue 값! 인자 하나가 더 생김!
        //현재 RMSprop clip값 = 0.5로 되어있음
        //SetOptimizer(new GradientDescentOptimizer<float>(GetParameter(), 0.001, 0.9, 1.0, MINIMIZE));                      // Optimizer의 첫번째 인자로 parameter목록을 전달해주는거고!!!   즉 updateparameter를 할 때 넘겨주는 parameter에 대해서만 함!!!!!
        SetOptimizer(new RMSPropOptimizer<float>(GetParameter(), 0.01, 0.9, 1e-08, FALSE, MINIMIZE));
        //SetOptimizer(new AdamOptimizer<float>(GetParameter(), 0.001, 0.9, 0.999, 1e-08, MINIMIZE));
        // SetOptimizer(new NagOptimizer<float>(GetParameter(), 0.001, 0.9, MINIMIZE));
        //GetParameter이거 호출했을 때 잘되는지 확인?   ㅇㅇ weight3개 잘 들어가 있음
        //std::cout<<"Getparameter 호출"<<'\n';
        //std::cout<<(*GetParameter())[0]->GetName()<<'\n';
        //std::cout<<(*GetParameter())[1]->GetName()<<'\n';
        //std::cout<<(*GetParameter())[2]->GetName()<<'\n';
        //std::cout<<(*GetParameter())[3]->GetName()<<'\n';
        //std::cout<<(*GetParameter())[4]->GetName()<<'\n';
        //SetOptimizer(new AdagradOptimizer<float>(GetParameter(), 0.001, 0.9, MINIMIZE));      //MAXIMIZE
    }

    virtual ~my_RNN() {}
};
