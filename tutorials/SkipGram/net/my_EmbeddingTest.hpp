#include <iostream>
#include <string>

#include "../../../WICWIU_src/NeuralNetwork.hpp"


class my_EmbeddingTest : public NeuralNetwork<float>{
private:
public:
    my_EmbeddingTest(Tensorholder<float> *x, Tensorholder<float> *label, Operator<float> *pWeight) {
        SetInput(2, x, label);

        Operator<float> *out = NULL;

        out = new EmbeddingTestLayer<float>(x, pWeight, "Embedding_Test_");

        AnalyzeGraph(out);

        // ======================= Select LossFunction Function ===================
        SetLossFunction(new NEG<float>(out, label, "NCE"));
        SetOptimizer(new AdagradOptimizer<float>(GetParameter(), 0.001, 0.9, MINIMIZE));

    }

    virtual ~my_EmbeddingTest() {}
};
