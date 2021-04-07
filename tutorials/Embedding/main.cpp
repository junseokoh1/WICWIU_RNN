#include "net/my_Embedding.hpp"
#include <time.h>
#include <unistd.h>
#include <iostream>
#include <string>
#include <fstream>   // ifstream 이게 파일 입력
#include <cstring>    //strlen 때문에 추가한 해더
#include <algorithm> //sort 때문에 추가한 헤더
// #include "TextDataset.hpp"
//#include "BatchTextDataSet.hpp"
//#include "WordTextDataSet.hpp"
#include "textData.hpp"

using namespace std;

#define BATCH                 1
#define EPOCH                 6
#define MAX_TRAIN_ITERATION    20000   // (60000 / BATCH)
#define MAX_TEST_ITERATION     100   // (10000 / BATCH)
#define GPUID                 2



int main(int argc, char const *argv[]) {

    clock_t startTime = 0, endTime = 0;
    double  nProcessExcuteTime = 0;

    //CBOW를 위해 추가
    textData<float> *dataset = new textData<float>("Data/debug.txt", 300, CBOWMODE);        //batch를 추가할까?????    이거 배열길이 주는거 중요함!!! 여기서 segmetation fault 뜸!
    //middlesize로 할때는 3500

    //int Text_length = dataset->GetTextLength();     //나머지는 이거!!!
    int word_num = dataset->GetWordNum();        //TextDataset2일때만 이거!!!! 중요!!!
    //int time_size = dataset->GetTimeSize();
    int vocab_length = dataset->GetVocabLength();

    int batchsize = word_num-2;                  //window 때문에 빼준거임

    std::cout<<"파일에 있는 단어 개수 : "<<word_num<<" / vocab 길이 : "<<vocab_length<<'\n';



    Tensorholder<float> *x_holder = new Tensorholder<float>(1, batchsize, 1, 1, vocab_length*2, "x");             //곱하기 2 해줘야하지!!!    window size!!!
    Tensorholder<float> *label_holder = new Tensorholder<float>(1, batchsize, 1, 1, vocab_length, "label");

    //batch를 적용했을 경우!
    //Tensorholder<float> *x_holder = new Tensorholder<float>(time_size, BATCH, 1, 1, vocab_length, "x");
    //Tensorholder<float> *label_holder = new Tensorholder<float>(time_size, BATCH, 1, 1, vocab_length, "label");

    NeuralNetwork<float> *net = new my_Embedding(x_holder, label_holder, vocab_length);

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

            net->Train();



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


        for (int j = 0; j < (int)MAX_TEST_ITERATION; j++) {

            net->Test();

            test_accuracy += net->GetAccuracy(vocab_length);
            test_avg_loss += net->GetLoss();

            printf("\rTest complete percentage is %d / %d -> loss : %f, acc : %f",
                   j + 1, MAX_TEST_ITERATION,
                   test_avg_loss / (j + 1),
                   test_accuracy / (j + 1));
            fflush(stdout);
        }

        std::cout << "\n\n";

    }

    delete net;

    return 0;
}
