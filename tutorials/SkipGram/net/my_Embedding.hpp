#include <iostream>
#include <string>

#include "../../../WICWIU_src/NeuralNetwork.hpp"


class my_Embedding : public NeuralNetwork<float>{
private:
public:
    my_Embedding(Tensorholder<float> *x, Tensorholder<float> *label, int vocab_length) {
        SetInput(2, x, label);

        Operator<float> *out = NULL;


        //CBOW 실험해보기!
        //out = new CBOWLayer<float>(x, vocab_length, 128, 2, "CBOW");
        out = new SKIPGRAMLayer<float>(x, vocab_length, 200, "SKIPGRAM");       // 128 : 이게 embedding dim!


        //skip gram with negative sampling에서는 sigmoid + corssentropy!!!
        out = new Sigmoid<float>(out, "sigmoid");

        AnalyzeGraph(out);

        // ======================= Select LossFunction Function ===================
        // SetLossFunction(new HingeLoss<float>(out, label, "HL"));
        //SetLossFunction(new MSE<float>(out, label, "MSE"));
        //SetLossFunction(new SoftmaxCrossEntropy<float>(out, label, "SCE"));                                     //이거 아니면 다른거임!
        //SetLossFunction(new CrossEntropy2<float>(out, label, "CE"));
        SetLossFunction(new NEG<float>(out, label, "NCE"));

        // ======================= Select Optimizer ===================
        //SetOptimizer(new GradientDescentOptimizer<float>(GetParameter(), 0.01, 0.9, 1.0, MINIMIZE));
        //SetOptimizer(new RMSPropOptimizer<float>(GetParameter(), 0.01, 0.9, 1e-08, FALSE, MINIMIZE));
        //SetOptimizer(new AdamOptimizer<float>(GetParameter(), 0.01, 0.9, 0.999, 1e-08, MINIMIZE));
        //SetOptimizer(new NagOptimizer<float>(GetParameter(), 0.001, 0.9, MINIMIZE));
        SetOptimizer(new AdagradOptimizer<float>(GetParameter(), 0.002, 0.9, MINIMIZE));                       // google의 코드에서는 0.025에서 시작해서 점점 감소
    }

    virtual ~my_Embedding() {}
};
