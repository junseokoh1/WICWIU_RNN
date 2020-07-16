#include "net/my_RNN.hpp"
#include <time.h>
#include <unistd.h>
#include <iostream>
#include <string>
#include <fstream>   // ifstream 이게 파일 입력
#include <cstring>    //strlen 때문에 추가한 해더
#include <algorithm> //sort 때문에 추가한 헤더
#include "TextDataset.hpp"
//#include "BatchTextDataSet.hpp"

using namespace std;

#define BATCH                 1
#define EPOCH                 6
#define MAX_TRAIN_ITERATION    20000   // (60000 / BATCH)
#define MAX_TEST_ITERATION     100   // (10000 / BATCH)
#define GPUID                 2



int main(int argc, char const *argv[]) {

    clock_t startTime = 0, endTime = 0;
    double  nProcessExcuteTime = 0;

    TextDataset<float> *dataset = new TextDataset<float>("Data/debug.txt", 100, ONEHOT);
    //batch 추가한 새로운 TextDataset!
    //BatchTextDataSet<float> *dataset = new BatchTextDataSet<float>("Data/test566.txt", BATCH, 100, ONEHOT);

    int Text_length = dataset->GetTextLength();
    //int time_size = dataset->GetTimeSize();
    int vocab_length = dataset->GetVocabLength();

    //std::cout<<"파일 길이 : "<<Text_length<<" vocab 길이 : "<<vocab_length<<'\n';



    Tensorholder<float> *x_holder = new Tensorholder<float>(Text_length, BATCH, 1, 1, vocab_length, "x");
    Tensorholder<float> *label_holder = new Tensorholder<float>(Text_length, BATCH, 1, 1, vocab_length, "label");

    //batch를 적용했을 경우!
    //Tensorholder<float> *x_holder = new Tensorholder<float>(time_size, BATCH, 1, 1, vocab_length, "x");
    //Tensorholder<float> *label_holder = new Tensorholder<float>(time_size, BATCH, 1, 1, vocab_length, "label");

    NeuralNetwork<float> *net = new my_RNN(x_holder,label_holder, vocab_length);

    //std::cout<<net->GetLossFunction()<<'\n';

#ifdef __CUDNN__
    std::cout<<"GPU환경에서 실행중 입니다."<<'\n';
    net->SetDeviceGPU(GPUID);
#endif  // __CUDNN__

    Tensor<float> *x = dataset->GetInputData();
    Tensor<float> *label = dataset->GetLabelData();

#ifdef __CUDNN__
            x->SetDeviceGPU(GPUID);
            label->SetDeviceGPU(GPUID);
#endif  // __CUDNN__

    //std::cout<<"입력 tensor값"<<'\n'<<x<<'\n';
    //std::cout<<"-----Label 값-----"<<'\n'<<label<<'\n';

    std::cout<<'\n';
    net->PrintGraphInformation();


    float best_acc = 0;
    int   epoch    = 0;

    net->FeedInputTensor(2, x, label);

    for (int i = epoch + 1; i < EPOCH; i++) {

        float train_accuracy = 0.f;
        float train_avg_loss = 0.f;

        //loss 값 확인할려고 내가 추가한 변수
        float temp_loss = 0.f;

        net->SetModeTrain();

        startTime = clock();

        //net->FeedInputTensor(2, x_tensor, label_tensor);                        //왜??? 왜 안에 넣어두면 안되는거지???

        // ============================== Train ===============================
        std::cout << "Start Train" <<'\n';

        for (int j = 0; j < MAX_TRAIN_ITERATION; j++) {


            // std::cin >> temp;
            //net->FeedInputTensor(2, x_tensor, label_tensor);                     //이 부분이 MNIST에서는 dataloader로 가져가서 이렇게 for문 안에 넣어둠
            net->ResetParameterGradient();

            //batch적용 전!
            //net->BPTT(Text_length);
            //truncated BPTT
            net->BPTT(Text_length);

            //batch로 했을 경우
            //net->BPTT(time_size);

            // std::cin >> temp;
            //train_accuracy += net->GetAccuracy(4);                               // default로는 10으로 되어있음   이게 기존꺼임
            //train_avg_loss += net->GetLoss();

            //Loss값 갑자기 커질때 확인
            //temp_loss = train_avg_loss;

            train_accuracy = net->GetAccuracy(vocab_length);
            train_avg_loss = net->GetLoss();

            //Loss값 갑자기 커질때 확인

          //  if( j !=0 && ((temp_loss - train_avg_loss)>2 || (train_avg_loss - temp_loss)>2)){
          //      std::cout<<'\n'<<"Loss값이 튐"<<'\n';
          //      std::cout<<"차이 :"<<temp_loss - train_avg_loss<<'\n';
          //      return 0;
          //  }


            //std::cout<<" 전달해준 loss값 : "<<net->GetLoss()<<'\n';

            printf("\rTrain complete percentage is %d / %d -> loss : %f, acc : %f"  ,
                   j + 1, MAX_TRAIN_ITERATION,
                   train_avg_loss, ///  (j + 1),                              //+=이니깐 j+1로 나눠주는거는 알겠는데........ 근데 왜 출력되는 값이 계속 작아지는 거지??? loss값이 같아도 왜 이건 작아지는거냐고...
                   train_accuracy  /// (j + 1)
                 );
            std::cout<<'\n';
            fflush(stdout);

        }

        endTime            = clock();
        nProcessExcuteTime = ((double)(endTime - startTime)) / CLOCKS_PER_SEC;
        printf("\n(excution time per epoch : %f)\n\n", nProcessExcuteTime);

        // ======================= Test ======================
        float test_accuracy = 0.f;
        float test_avg_loss = 0.f;

        net->SetModeInference();

        std::cout << "Start Test" <<'\n';

        //net->FeedInputTensor(2, x_tensor, label_tensor);

        for (int j = 0; j < (int)MAX_TEST_ITERATION; j++) {
            // #ifdef __CUDNN__
            //         x_t->SetDeviceGPU(GPUID);
            //         l_t->SetDeviceGPU(GPUID);
            // #endif  // __CUDNN__

            //net->FeedInputTensor(2, x_tensor, label_tensor);
            net->BPTT_Test(Text_length);

            test_accuracy += net->GetAccuracy(vocab_length);
            test_avg_loss += net->GetLoss();

            printf("\rTest complete percentage is %d / %d -> loss : %f, acc : %f",
                   j + 1, MAX_TEST_ITERATION,
                   test_avg_loss / (j + 1),
                   test_accuracy / (j + 1));
            fflush(stdout);
        }

        std::cout << "\n\n";

    }       // 여기까지가 epoc for문

    delete net;

    return 0;
}
