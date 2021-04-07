#ifndef __MODULE_UTIL_H__
#define __MODULE_UTIL_H__    value

#include "Module/LinearLayer.hpp"
#include "Module/ConvolutionLayer.hpp"
#include "Module/BatchNormalizeLayer.hpp"
// #include "Module/CUDNNBatchNormalizeLayer.h"
#include "Module/TransposedConvolutionLayer.hpp"
#include "Module/RecurrentLayer.hpp"
#include "Module/DeepRecurrentLayer.hpp"
#include "Module/LSTMLayer.hpp"
#include "Module/LSTM2Layer.hpp"
#include "Module/GRULayer.hpp"

#include "Module/CBOWLayer.hpp"
#include "Module/SKIPGRAMLayer.hpp"
#include "Module/EmbeddingTestLayer.hpp"
#include "Module/EmbeddingLayer.hpp"

#include "Module/Encoder.hpp"
#include "Module/Decoder.hpp"
#include "Module/Decoder2.hpp"

//attention을 위해 추가
#include "Module/AttentionWeight.hpp"
#include "Module/AttentionModule.hpp"
#include "Module/AttentionDecoder_Module.hpp"

#endif // ifndef __MODULE_UTIL_H__
