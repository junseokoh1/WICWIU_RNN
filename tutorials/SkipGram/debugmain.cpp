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

#define BATCH                 2
#define EPOCH                 2
#define LOOP_FOR_TRAIN        2   // (60000 / BATCH)
#define MAX_TEST_ITERATION    5   // (10000 / BATCH)
#define GPUID                 2

#define WINDOW                5
#define NEGATIVE              3



int main(int argc, char const *argv[]) {

    clock_t startTime = 0, endTime = 0;
    double  nProcessExcuteTime = 0;

    int vocab_size  = 10;



    Tensorholder<float> *x_holder = new Tensorholder<float>(1, BATCH, 1, 1, 4, "x");
    Tensorholder<float> *label_holder = new Tensorholder<float>(1, BATCH, 1, 1, 3, "label");

    NeuralNetwork<float> *net = new my_Embedding(x_holder, label_holder, vocab_size);


  net->PrintGraphInformation();

  float best_acc = 0;
  int   epoch    = 0;


  for (int i = epoch + 1; i < EPOCH; i++) {
      std::cout << "EPOCH : " << i << '\n';

      // ======================= Train =======================
      float train_accuracy = 0.f;
      float train_avg_loss = 0.f;

      net->SetModeTrain();

      startTime = clock();

      for (int j = 0; j < LOOP_FOR_TRAIN; j++) {

          /*

            입력
            batch 0 : [2, 1, 5, 8]
            batch 1 : [2, 3, 6, 7]

          */

          Tensor<float> *x_t = new Tensor<float>(1, BATCH, 1, 1, 4);
          Tensor<float> *l_t = new Tensor<float>(1, BATCH, 1, 1, 3);

          // x_t[0] = 2;  x_t[1] = 1;  x_t[2] = 5;  x_t[3] = 8;
          // x_t[4] = 2;  x_t[5] = 3;  x_t[6] = 6;  x_t[7] = 7;
          //
          // l_t[0] = 1;  l_t[1] = 0;  l_t[2] = 0;
          // l_t[3] = 1;  l_t[4] = 0;  l_t[5] = 0;

          (*x_t)[Index5D(x_t->GetShape(), 0, 0, 0, 0, 0)] = 2; (*x_t)[Index5D(x_t->GetShape(), 0, 0, 0, 0, 1)] = 1; (*x_t)[Index5D(x_t->GetShape(), 0, 0, 0, 0, 2)] = 5; (*x_t)[Index5D(x_t->GetShape(), 0, 0, 0, 0, 3)] = 8;
          (*x_t)[Index5D(x_t->GetShape(), 0, 1, 0, 0, 0)] = 2; (*x_t)[Index5D(x_t->GetShape(), 0, 1, 0, 0, 1)] = 3; (*x_t)[Index5D(x_t->GetShape(), 0, 1, 0, 0, 2)] = 6; (*x_t)[Index5D(x_t->GetShape(), 0, 1, 0, 0, 3)] = 7;


          (*l_t)[Index5D(l_t->GetShape(), 0, 0, 0, 0, 0)] = 1; (*l_t)[Index5D(l_t->GetShape(), 0, 0, 0, 0, 1)] = 0; (*l_t)[Index5D(l_t->GetShape(), 0, 0, 0, 0, 2)] = 0;
          (*l_t)[Index5D(l_t->GetShape(), 0, 1, 0, 0, 0)] = 1; (*l_t)[Index5D(l_t->GetShape(), 0, 1, 0, 0, 1)] = 0; (*l_t)[Index5D(l_t->GetShape(), 0, 1, 0, 0, 2)] = 0;

          //입력 값을 잘 만들어 주는지 확인!!
          // std::cout<<i<<"번째 입력의 값"<<'\n';
          // std::cout<<x_t->GetShape()<<'\n';
          std::cout<<x_t<<'\n';
          //
          // std::cout<<i<<"번째 Label의 값"<<'\n';
          // std::cout<<l_t->GetShape()<<'\n';
          std::cout<<l_t<<'\n';


          // std::cin >> temp;
          net->FeedInputTensor(2, x_t, l_t);
          net->ResetParameterGradient();
          net->Train();

          // std::cin >> temp;

          train_accuracy += net->GetAccuracy(NEGATIVE+1);
          train_avg_loss = net->GetLoss();
          //std::cout<<"갖고오는 loss 값 : "<<net->GetLoss()<<'\n';

          printf("\rTrain complete percentage is %d / %d -> loss : %f, acc : %f"  /*(ExcuteTime : %f)*/,
                 j + 1, LOOP_FOR_TRAIN,
                 train_avg_loss ,/// (j + 1),
                 train_accuracy / (j + 1)
                 /*nProcessExcuteTime*/);
          fflush(stdout);

      }


    }

    delete net;

    return 0;
}
