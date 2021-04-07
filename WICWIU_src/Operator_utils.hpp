#ifndef __OPERATER_UTIL_H__
#define __OPERATER_UTIL_H__    value

#include "Operator/Tensorholder.hpp"
#include "Operator/ReShape.hpp"
#include "Operator/Concatenate.hpp"

#include "Operator/Relu.hpp"
#include "Operator/LRelu.hpp"
#include "Operator/PRelu.hpp"
#include "Operator/Sigmoid.hpp"
#include "Operator/Tanh.hpp"

#include "Operator/Add.hpp"
#include "Operator/MatMul.hpp"

#include "Operator/Convolution.hpp"
#include "Operator/TransposedConvolution.hpp"
#include "Operator/Maxpooling.hpp"
#include "Operator/Avgpooling.hpp"

#include "Operator/BatchNormalize.hpp"
// #include "Operator/CUDNNBatchNormalize.h"

#include "Operator/Softmax.hpp"
// #include "Operator/Dropout.h"

#include "Operator/NoiseGenerator/GaussianNoiseGenerator.hpp"
#include "Operator/NoiseGenerator/UniformNoiseGenerator.hpp"

#include "Operator/Switch.hpp"
#include "Operator/ReconstructionError.hpp"

//RNN관련하여 추가
#include "Operator/Recurrent.hpp"
#include "Operator/Hadamard.hpp"
#include "Operator/LSTM.hpp"
#include "Operator/LSTM2.hpp"
#include "Operator/SeqLSTM2.hpp"
#include "Operator/Minus.hpp"
#include "Operator/GRU.hpp"
#include "Operator/SeqGRU.hpp"
//#include "Operator/Minus.hpp"
#include "Operator/RecurrentCUDNN.hpp"
#include "Operator/RecurrentCUDNN2.hpp"

//embedding 관련해서 추가됨
#include "Operator/CBOW.hpp"
#include "Operator/Embedding.hpp"             //순서도 영향이 있음!!!!
#include "Operator/CBOWEmbedding.hpp"

#include "Operator/DotProduct.hpp"
#include "Operator/SkipGram.hpp"
#include "Operator/EmbeddingTest.hpp"

//encoder, decoder 관련해서 추가
#include "Operator/SeqRecurrent.hpp"

#include "Operator/Attention.hpp"
#include "Operator/DotAttention.hpp"

//2021/03/13
#include "Operator/DotSimilarity.hpp"
#include "Operator/AttentionByModule.hpp"

#include "Operator/MaskedFill.hpp"
#include "Operator/PaddingAttentionMaskRNN.hpp"

//RNN 빠르게 수정
#include "Operator/HiddenMatMul.hpp"
//#include "Operator/FastRecurrent.hpp"

#endif  // __OPERATER_UTIL_H__
