#include "net/my_Embedding.hpp"
#include "net/my_EmbeddingTest.hpp"
#include <time.h>
#include <unistd.h>
#include <iostream>
#include <string>
#include <fstream>   // ifstream 이게 파일 입력
#include <cstring>    //strlen 때문에 추가한 해더
#include <algorithm> //sort 때문에 추가한 헤더

#include "text8.hpp"
#include "accuracy.hpp"

using namespace std;

#define BATCH                 22
#define EPOCH                 10
#define LOOP_FOR_TRAIN        100   // (60000 / BATCH)
#define MAX_TEST_ITERATION    10   // (10000 / BATCH)
#define GPUID                 2

#define WINDOW                5
#define NEGATIVE              3



int main(int argc, char const *argv[]) {

    clock_t startTime = 0, endTime = 0;
    double  nProcessExcuteTime = 0;

    //text8<float> *train_dataset = new text8<float>("Data/debug.txt", WINDOW, NEGATIVE, SKIPGRAM);
    text8<float> *train_dataset = new text8<float>("Data/subtext8-2.txt", WINDOW, NEGATIVE, SKIPGRAM);
    DataLoader<float> * train_dataloader = new DataLoader<float>(train_dataset, BATCH, TRUE, 20, FALSE);

    accuracy<float> *test_dataset = new accuracy<float>("Data/questions-words.txt", train_dataset->GetVocab(), train_dataset->GetVocabSize());
    DataLoader<float> * test_dataloader = new DataLoader<float>(test_dataset, BATCH, TRUE, 20, FALSE);

    int word_num      = train_dataset->GetWordNum();
    int vocab_size  = train_dataset->GetVocabSize();

    std::cout<<"Train 파일에 있는 단어 개수 : "<<word_num<<" / vocab 개수 : "<<vocab_size<<'\n';

    int inputDim      = train_dataset->GetInputDim();
    int LabelDim      = train_dataset->GetLabelDim();

    Tensorholder<float> *x_holder = new Tensorholder<float>(1, BATCH, 1, 1, inputDim, "x");
    Tensorholder<float> *label_holder = new Tensorholder<float>(1, BATCH, 1, 1, LabelDim, "label");

    NeuralNetwork<float> *net = new my_Embedding(x_holder, label_holder, vocab_size);

    //weight값을 가져올 수 있는지 확인하기
    // std::cout<<"Parameter 이름"<<'\n';
    // std::cout<<(*net->GetParameter())[0]->GetName()<<'\n';
    // std::cout<<(*net->GetParameter())[0]->GetResult()<<'\n';


#ifdef __CUDNN__
    std::cout<<"GPU환경에서 실행중 입니다."<<'\n';
    net->SetDeviceGPU(GPUID);
#endif  // __CUDNN__


#ifdef __CUDNN__
            x->SetDeviceGPU(GPUID);
            label->SetDeviceGPU(GPUID);
#endif  // __CUDNN__


net->PrintGraphInformation();

float best_acc = 0;
int   epoch    = 0;

std::cout << "best_acc : " << best_acc << '\n';
std::cout << "epoch : " << epoch << '\n';

for (int i = epoch + 1; i < EPOCH; i++) {
    std::cout << "EPOCH : " << i << '\n';

    if ((i + 1) % 50 == 0) {
        std::cout << "Change learning rate!" << '\n';
        float lr = net->GetOptimizer()->GetLearningRate();
        net->GetOptimizer()->SetLearningRate(lr * 0.1);
    }

    // ======================= Train =======================
    float train_accuracy = 0.f;
    float train_avg_loss = 0.f;

    net->SetModeTrain();

    startTime = clock();

    for (int j = 0; j < LOOP_FOR_TRAIN; j++) {
        //dataset->CreateTrainDataPair(BATCH);
        std::vector<Tensor<float> *> * temp =  train_dataloader->GetDataFromGlobalBuffer();
        // printf("%d\r\n", temp->size());

        Tensor<float> *x_t = (*temp)[0];
        Tensor<float> *l_t = (*temp)[1];
        delete temp;

        //입력 값을 잘 만들어 주는지 확인!!
         // std::cout<<'\n'<<i<<"번째 입력의 값"<<'\n';
         // std::cout<<x_t->GetShape()<<'\n';
         // std::cout<<x_t<<'\n';
        //
        // std::cout<<i<<"번째 Label의 값"<<'\n';
        // std::cout<<l_t->GetShape()<<'\n';
        // std::cout<<l_t<<'\n';


#ifdef __CUDNN__
        x_t->SetDeviceGPU(GPUID);
        l_t->SetDeviceGPU(GPUID);
#endif  // __CUDNN__
        // std::cin >> temp;
        net->FeedInputTensor(2, x_t, l_t);
        net->ResetParameterGradient();
        net->Train();

        // std::cin >> temp;

        //train_accuracy += net->GetAccuracy(NEGATIVE+1);
        train_avg_loss = net->GetLoss();
        //std::cout<<"갖고오는 loss 값 : "<<net->GetLoss()<<'\n';

        printf("\rTrain complete percentage is %d / %d -> loss : %f"  /*(ExcuteTime : %f)*/,
               j + 1, LOOP_FOR_TRAIN,
               train_avg_loss //,/ (j + 1),
               //train_accuracy / (j + 1)
               /*nProcessExcuteTime*/);
        fflush(stdout);

        if(j%3000 == 0)
            std::cout<<'\n';
    }

    endTime            = clock();
    nProcessExcuteTime = ((double)(endTime - startTime)) / CLOCKS_PER_SEC;
    printf("\n(excution time per epoch : %f)\n\n", nProcessExcuteTime);


// ============================================================================================ Test ===============================================================================================
        float test_accuracy = 0.f;
        float test_avg_loss = 0.f;

        //test task는 입력의 shape = 3, label = 1 로 정해져있지!!!
        x_holder = new Tensorholder<float>(1, BATCH, 1, 1, 3, "x");
        label_holder = new Tensorholder<float>(1, BATCH, 1, 1, 1, "label");

        // std::cout<<"넘겨주는 weight의 이름"<<'\n'<<(*net->GetParameter())[0]->GetName()<<'\n';

        NeuralNetwork<float> *testNet = new my_EmbeddingTest( x_holder, label_holder, (*net->GetParameter())[0] );

        std::cout << "Start Test" <<'\n';

        for (int j = 0; j < (int)MAX_TEST_ITERATION; j++) {

          std::vector<Tensor<float> *> * temp =  test_dataloader->GetDataFromGlobalBuffer();
          Tensor<float> *x_t = (*temp)[0];
          Tensor<float> *l_t = (*temp)[1];
          delete temp;

          //입력 값을 잘 만들어 주는지 확인!!
          // std::cout<<i<<"번째 입력의 값"<<'\n';
          // std::cout<<x_t->GetShape()<<'\n';
          // std::cout<<x_t<<'\n';
          //
          // std::cout<<i<<"번째 Label의 값"<<'\n';
          // std::cout<<l_t->GetShape()<<'\n';
          // std::cout<<l_t<<'\n';

#ifdef __CUDNN__
          x_t->SetDeviceGPU(GPUID);
          l_t->SetDeviceGPU(GPUID);
#endif  // __CUDNN__

            testNet->FeedInputTensor(2, x_t, l_t);

            testNet->Test();                                            //이거.... 음... lossfunction forward하는 부분이 필요없는데.... 허허허

            test_accuracy += testNet->GetIndexAccuracy();

            //뭘 결과로 갖고왔는지 확인해보자....!!!
            testNet->GetEmbeddingResult(train_dataset->GetVocab());

            printf("\rTest complete percentage is %d / %d -> acc : %f",
                   j + 1, MAX_TEST_ITERATION,
                   test_accuracy / (j + 1) );
            fflush(stdout);
        }

        std::cout << "\n\n";

    }

    delete net;

    return 0;
}
