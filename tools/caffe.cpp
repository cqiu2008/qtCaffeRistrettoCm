#ifdef WITH_PYTHON_LAYER
#include "boost/python.hpp"
namespace bp = boost::python;
#endif

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "caffe/caffe.hpp"
#include "caffe/util/signal_handler.h"

//#include <sys/stat.h>
//#include <sys/types.h>

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::Solver;
using caffe::shared_ptr;
using caffe::string;
using caffe::Timer;
using caffe::vector;
using std::ostringstream;

DEFINE_string(gpu, "",
    "Optional; run in GPU mode on given device IDs separated by ','."
    "Use '-gpu all' to run on all available GPUs. The effective training "
    "batch size is multiplied by the number of devices.");
DEFINE_string(solver, "",
    "The solver definition protocol buffer text file.");
DEFINE_string(model, "",
    "The model definition protocol buffer text file.");
DEFINE_string(snapshot, "",
    "Optional; the snapshot solver state to resume training.");
DEFINE_string(weights, "",
    "Optional; the pretrained weights to initialize finetuning, "
    "separated by ','. Cannot be set simultaneously with snapshot.");
DEFINE_int32(iterations, 50,
    "The number of iterations to run.");
DEFINE_string(sigint_effect, "stop",
             "Optional; action to take when a SIGINT signal is received: "
              "snapshot, stop or none.");
DEFINE_string(sighup_effect, "snapshot",
             "Optional; action to take when a SIGHUP signal is received: "
             "snapshot, stop or none.");

// A simple registry for caffe commands.
typedef int (*BrewFunction)();
typedef std::map<caffe::string, BrewFunction> BrewMap;
BrewMap g_brew_map;

#define RegisterBrewFunction(func) \
namespace { \
class __Registerer_##func { \
 public: /* NOLINT */ \
  __Registerer_##func() { \
    g_brew_map[#func] = &func; \
  } \
}; \
__Registerer_##func g_registerer_##func; \
}

static BrewFunction GetBrewFunction(const caffe::string& name) {
  if (g_brew_map.count(name)) {
    return g_brew_map[name];
  } else {
    LOG(ERROR) << "Available caffe actions:";
    for (BrewMap::iterator it = g_brew_map.begin();
         it != g_brew_map.end(); ++it) {
      LOG(ERROR) << "\t" << it->first;
    }
    LOG(FATAL) << "Unknown action: " << name;
    return NULL;  // not reachable, just to suppress old compiler warnings.
  }
}

void CoutBlobShape(vector<int> &shape)
{
	std::cout<<"(";
	for(int j = 0; j < shape.size(); j++)
	{
		if (j!= 0)
		{
		  std::cout<<",";
		}
		std::cout<<shape[j];
	}
	std::cout<<")"<<std::endl;
}

bool matchString(std::string& str_tmp,
                 std::string str_m1,
                 std::string str_m2,
                 std::string str_m_no){

    bool match1 = str_tmp.npos != str_tmp.find(str_m1);
    bool match2 = str_tmp.npos != str_tmp.find(str_m2);
    bool not_match = str_tmp.npos == str_tmp.find(str_m_no);
    if( (match1 || match2) && not_match){
        return true;
    }else{
        return false;
    }
}





// Parse GPU ids or use all available devices
static void get_gpus(vector<int>* gpus) {
  if (FLAGS_gpu == "all") {
    int count = 0;
#ifndef CPU_ONLY
    CUDA_CHECK(cudaGetDeviceCount(&count));
#else
    NO_GPU;
#endif
    for (int i = 0; i < count; ++i) {
      gpus->push_back(i);
    }
  } else if (FLAGS_gpu.size()) {
    vector<string> strings;
    boost::split(strings, FLAGS_gpu, boost::is_any_of(","));
    for (int i = 0; i < strings.size(); ++i) {
      gpus->push_back(boost::lexical_cast<int>(strings[i]));
    }
  } else {
    CHECK_EQ(gpus->size(), 0);
  }
}

// caffe commands to call by
//     caffe <command> <args>
//
// To add a command, define a function "int command()" and register it with
// RegisterBrewFunction(action);

// Device Query: show diagnostic information for a GPU device.
int device_query() {
  LOG(INFO) << "Querying GPUs " << FLAGS_gpu;
  vector<int> gpus;
  get_gpus(&gpus);
  for (int i = 0; i < gpus.size(); ++i) {
    caffe::Caffe::SetDevice(gpus[i]);
    caffe::Caffe::DeviceQuery();
  }
  return 0;
}
RegisterBrewFunction(device_query);

// Load the weights from the specified caffemodel(s) into the train and
// test nets.
void CopyLayers(caffe::Solver<float>* solver, const std::string& model_list) {
  std::vector<std::string> model_names;
  boost::split(model_names, model_list, boost::is_any_of(",") );
  for (int i = 0; i < model_names.size(); ++i) {
    LOG(INFO) << "Finetuning from " << model_names[i];
    solver->net()->CopyTrainedLayersFrom(model_names[i]);
    for (int j = 0; j < solver->test_nets().size(); ++j) {
      solver->test_nets()[j]->CopyTrainedLayersFrom(model_names[i]);
    }
  }
}

// Translate the signal effect the user specified on the command-line to the
// corresponding enumeration.
caffe::SolverAction::Enum GetRequestedAction(
    const std::string& flag_value) {
  if (flag_value == "stop") {
    return caffe::SolverAction::STOP;
  }
  if (flag_value == "snapshot") {
    return caffe::SolverAction::SNAPSHOT;
  }
  if (flag_value == "none") {
    return caffe::SolverAction::NONE;
  }
  LOG(FATAL) << "Invalid signal effect \""<< flag_value << "\" was specified";
}

// Train / Finetune a model.
int train() {
  CHECK_GT(FLAGS_solver.size(), 0) << "Need a solver definition to train.";
  CHECK(!FLAGS_snapshot.size() || !FLAGS_weights.size())
      << "Give a snapshot to resume training or weights to finetune "
      "but not both.";

  caffe::SolverParameter solver_param;
  caffe::ReadSolverParamsFromTextFileOrDie(FLAGS_solver, &solver_param);

  // If the gpus flag is not provided, allow the mode and device to be set
  // in the solver prototxt.
  if (FLAGS_gpu.size() == 0
      && solver_param.solver_mode() == caffe::SolverParameter_SolverMode_GPU) {
      if (solver_param.has_device_id()) {
          FLAGS_gpu = "" +
              boost::lexical_cast<string>(solver_param.device_id());
      } else {  // Set default GPU if unspecified
          FLAGS_gpu = "" + boost::lexical_cast<string>(0);
      }
  }

  vector<int> gpus;
  get_gpus(&gpus);
  if (gpus.size() == 0) {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  } else {
    ostringstream s;
    for (int i = 0; i < gpus.size(); ++i) {
      s << (i ? ", " : "") << gpus[i];
    }
    LOG(INFO) << "Using GPUs " << s.str();
#ifndef CPU_ONLY
    cudaDeviceProp device_prop;
    for (int i = 0; i < gpus.size(); ++i) {
      cudaGetDeviceProperties(&device_prop, gpus[i]);
      LOG(INFO) << "GPU " << gpus[i] << ": " << device_prop.name;
    }
#endif
    solver_param.set_device_id(gpus[0]);
    Caffe::SetDevice(gpus[0]);
    Caffe::set_mode(Caffe::GPU);
    Caffe::set_solver_count(gpus.size());
  }

  caffe::SignalHandler signal_handler(
        GetRequestedAction(FLAGS_sigint_effect),
        GetRequestedAction(FLAGS_sighup_effect));

  shared_ptr<caffe::Solver<float> >
      solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));

  solver->SetActionFunction(signal_handler.GetActionFunction());

  if (FLAGS_snapshot.size()) {
    LOG(INFO) << "Resuming from " << FLAGS_snapshot;
    solver->Restore(FLAGS_snapshot.c_str());
  } else if (FLAGS_weights.size()) {
    CopyLayers(solver.get(), FLAGS_weights);
  }

  if (gpus.size() > 1) {
    caffe::P2PSync<float> sync(solver, NULL, solver->param());
    sync.Run(gpus);
  } else {
    LOG(INFO) << "Starting Optimization";
    solver->Solve();
  }
  LOG(INFO) << "Optimization Done.";
  return 0;
}
RegisterBrewFunction(train);

void logCaffeLayer(Net<float>& caffe_net){
    ////////==== debugPrint Begin
    //#if CAFFE_EN1 // by cqiu
    ////////Printf FeatureMap
     vector<string> layers_names = caffe_net.layer_names();
     vector<shared_ptr<Layer<float> > > layers = caffe_net.layers();
     string strName;
     FILE *oFile;
     char datFileName[1024]={0};
     char strNameChar[512]={0};
     for(int i = 0; i < layers_names.size(); i++){
       float maxData=-1024;
       float minData=1023;
        std::cout<<layers_names[i]<<std::endl;
        strName = layers_names[i];
        strcpy(strNameChar,strName.c_str());
        //if( (matchString(strName,"Conv","Elt","ReLU")) ){
        if( (matchString(strName,"conv","fc","ReLU")) ){////For alex net
          Layer<float>* layer = layers[i].get();
          caffe::LayerParameter layer_param = layer->layer_param();
          int bw_layer_in     = layer_param.quantization_param().bw_layer_in();
          int bw_layer_out    = layer_param.quantization_param().bw_layer_out();
          int bw_params_      = layer_param.quantization_param().bw_params();
          int fl_layer_in_    = layer_param.quantization_param().fl_layer_in();
          int fl_layer_out_   = layer_param.quantization_param().fl_layer_out();
          int fl_params_      = layer_param.quantization_param().fl_params();
          int fl_params_bias_ = layer_param.quantization_param().fl_params_bias();
          std::cout<<strName<<"="<<i<<std::endl;
          std::cout<<"bw_layer_in     ="<< bw_layer_in<<std::endl;
          std::cout<<"bw_layer_out    ="<< bw_layer_out<<std::endl;
          std::cout<<"bw_params_      ="<< bw_params_<<std::endl;
          std::cout<<"fl_layer_in_    ="<< fl_layer_in_<<std::endl;
          std::cout<<"fl_layer_out_   ="<< fl_layer_out_<<std::endl;
          std::cout<<"fl_params_      ="<< fl_params_<<std::endl;
          std::cout<<"fl_params_bias_ ="<< fl_params_bias_<<std::endl;
          char *strNamePt = strNameChar;
          while(*strNamePt){
              if(*strNamePt == '/'){
                *strNamePt = '-';
              }
              strNamePt++;
          }
          sprintf(datFileName,"cnnData/Layer/%s.dat",strNameChar);
          oFile = fopen(datFileName, "w+");
          Blob<float>* featureMap = caffe_net.getBlobByName(strName);
          const float* outPutFeature = featureMap->cpu_data();
          float dataTmp;
          for (int j=0; j < featureMap->height()*featureMap->width(); j++){
              for (int i = 0; i < featureMap->channels(); i++){
                  if(maxData < outPutFeature[i*featureMap->height()*featureMap->width()+j]){
                      maxData = outPutFeature[i*featureMap->height()*featureMap->width()+j];
                  }
                  if(minData > outPutFeature[i*featureMap->height()*featureMap->width()+j]){
                      minData = outPutFeature[i*featureMap->height()*featureMap->width()+j];
                  }
                  //====height->width->channels
                  dataTmp =outPutFeature[i*featureMap->height()*featureMap->width()+j];
                  if(  (0.00001f > dataTmp) && (-0.00001f < dataTmp)){ /// only print 0.00f,no -0.000f
                      dataTmp = 0 ;
                  }
                  //fprintf(oFile,"%4f\n",float(dataTmp*(1<<fl_layer_out_)));
                  if(fl_layer_out_ >= 16){
                      fl_layer_out_ = 4;
                  }
                  if(fl_layer_out_ >0){
                    fprintf(oFile,"%4d\n",int(dataTmp*(1<<fl_layer_out_)));
                  }else{
                    fprintf(oFile,"%4d\n",int(dataTmp/(1<<(-fl_layer_out_))));
                  }
              }
          }
          printf("minData,maxData=%f,%f\n",minData,maxData);
          fclose(oFile);
        }
     }
    ////////==== debugPrint End
}
void logCaffeBlob(Net<float>& caffe_net){
    ////////==== debugPrint Begin
    ////////Printf FeatureMap
     vector<string> blobs_names = caffe_net.blob_names();
     vector<shared_ptr<Blob<float> > > blobs = caffe_net.blobs();
     string strName;
  //   std::ofstream oFileData;
     FILE *oFile;
     char datFileName[1024]={0};
     char strNameChar[512]={0};
     for(int i = 0; i < blobs_names.size(); i++){
       float maxData=-1024;
       float minData=1023;
        std::cout<<blobs_names[i];
        vector<int> shape = blobs[i].get()->shape();
        CoutBlobShape(shape);
        strName = blobs_names[i];
        strcpy(strNameChar,strName.c_str());
        char *strNamePt = strNameChar;
        while(*strNamePt){
            if(*strNamePt == '/'){
                *strNamePt = '-';
            }
            strNamePt++;
        }
        sprintf(datFileName,"cnnData/Blob/%s.dat",strNameChar);
        oFile = fopen(datFileName, "w+");
        Blob<float>* featureBlob = caffe_net.getBlobByName(strName);
        const float* outPutFeature = featureBlob->cpu_data();
        float dataTmp;
        for (int j=0; j < featureBlob->height()*featureBlob->width(); j++){
            for (int i = 0; i < featureBlob->channels(); i++){
              if(maxData < outPutFeature[i*featureBlob->height()*featureBlob->width()+j]){
                  maxData = outPutFeature[i*featureBlob->height()*featureBlob->width()+j];
              }
              if(minData > outPutFeature[i*featureBlob->height()*featureBlob->width()+j]){
                  minData = outPutFeature[i*featureBlob->height()*featureBlob->width()+j];
              }
              dataTmp =outPutFeature[i*featureBlob->height()*featureBlob->width()+j];
              if(  (0.00001f > dataTmp) && (-0.00001f < dataTmp)){ /// only print 0.00f,no -0.000f
                  dataTmp = 0 ; }
              fprintf(oFile,"%4f\n",float(dataTmp));
          }
       }
        printf("minData,maxData=%f,%f\n",minData,maxData);
        fclose(oFile);
     }
}
// Test: score a model.
int test() {
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to score.";
  CHECK_GT(FLAGS_weights.size(), 0) << "Need model weights to score.";
  system("rm -rf cnnData");
  system("cp -rf emptyCnn cnnData");
  // Set device id and mode
  vector<int> gpus;
  get_gpus(&gpus);
  if (gpus.size() != 0) {
    LOG(INFO) << "Use GPU with device ID " << gpus[0];
#ifndef CPU_ONLY
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, gpus[0]);
    LOG(INFO) << "GPU device name: " << device_prop.name;
#endif
    Caffe::SetDevice(gpus[0]);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
  // Instantiate the caffe net.
  Net<float> caffe_net(FLAGS_model, caffe::TEST);
  caffe_net.CopyTrainedLayersFrom(FLAGS_weights);
  LOG(INFO) << "Running for " << FLAGS_iterations << " iterations.";

  vector<int> test_score_output_id;
  vector<float> test_score;
  float loss = 0;
  for (int i = 0; i < FLAGS_iterations; ++i) {
    float iter_loss;
    const vector<Blob<float>*>& result =
        caffe_net.Forward(&iter_loss);
    loss += iter_loss;
    int idx = 0;
    for (int j = 0; j < result.size(); ++j) {
      const float* result_vec = result[j]->cpu_data();
      for (int k = 0; k < result[j]->count(); ++k, ++idx) {
        const float score = result_vec[k];
        if (i == 0) {
          test_score.push_back(score);
          test_score_output_id.push_back(j);
        } else {
          test_score[idx] += score;
        }
        const std::string& output_name = caffe_net.blob_names()[
            caffe_net.output_blob_indices()[j]];
        LOG(INFO) << "Batch " << i << ", " << output_name << " = " << score;
      }
    }
  }
  loss /= FLAGS_iterations;
  LOG(INFO) << "Loss: " << loss;
  float maxScore=-16383;
  int maxIndex=0;
  for (int i = 0; i < test_score.size(); ++i) {
    const std::string& output_name = caffe_net.blob_names()[
        caffe_net.output_blob_indices()[test_score_output_id[i]]];
    const float loss_weight = caffe_net.blob_loss_weights()[
        caffe_net.output_blob_indices()[test_score_output_id[i]]];
    std::ostringstream loss_msg_stream;
    const float mean_score = test_score[i] / FLAGS_iterations;
    if (loss_weight) {
      loss_msg_stream << " (* " << loss_weight
                      << " = " << loss_weight * mean_score << " loss)";
    }
    if(maxScore < mean_score){
        maxScore = mean_score;
        maxIndex = i;
    }
    LOG(INFO) << output_name << " = " << mean_score << loss_msg_stream.str();
  }
    LOG(INFO) <<"Max Index = " << maxIndex <<" Max Value  = " << maxScore;


  ////////==== debugPrint Begin
  //#if CAFFE_EN1 // by cqiu
  #if 1// by cqiu
    logCaffeBlob(caffe_net);
    logCaffeLayer(caffe_net);
  #endif
  ////////==== debugPrint End


	////Print the result Add by cqiu  Begin
	/* Copy the output layer to a std::vector */
#if 0
	Blob<float>* output_layer = caffe_net->output_blobs()[0];
	const float* begin = output_layer->cpu_data();
	const float* end = begin + output_layer->channels();

	std::vector<float> output = std::vector<float>(begin, end);

	  N = std::min<int>(labels_.size(), N);
	  std::vector<int> maxN = Argmax(output, N);
	  std::vector<Prediction> predictions;
	  for (int i = 0; i < N; ++i) {
	    int idx = maxN[i];
	    predictions.push_back(std::make_pair(labels_[idx], output[idx]));
	  }

	  return predictions;
#endif

	////Print the result Add by cqiu  End



  return 0;
}
RegisterBrewFunction(test);


// Time: benchmark the execution time of a model.
int time() {
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to time.";

  // Set device id and mode
  vector<int> gpus;
  get_gpus(&gpus);
  if (gpus.size() != 0) {
    LOG(INFO) << "Use GPU with device ID " << gpus[0];
    Caffe::SetDevice(gpus[0]);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
  // Instantiate the caffe net.
  Net<float> caffe_net(FLAGS_model, caffe::TRAIN);

  // Do a clean forward and backward pass, so that memory allocation are done
  // and future iterations will be more stable.
  LOG(INFO) << "Performing Forward";
  // Note that for the speed benchmark, we will assume that the network does
  // not take any input blobs.
  float initial_loss;
  caffe_net.Forward(&initial_loss);
  LOG(INFO) << "Initial loss: " << initial_loss;
  LOG(INFO) << "Performing Backward";
  caffe_net.Backward();

  const vector<shared_ptr<Layer<float> > >& layers = caffe_net.layers();
  const vector<vector<Blob<float>*> >& bottom_vecs = caffe_net.bottom_vecs();
  const vector<vector<Blob<float>*> >& top_vecs = caffe_net.top_vecs();
  const vector<vector<bool> >& bottom_need_backward =
      caffe_net.bottom_need_backward();
  LOG(INFO) << "*** Benchmark begins ***";
  LOG(INFO) << "Testing for " << FLAGS_iterations << " iterations.";
  Timer total_timer;
  total_timer.Start();
  Timer forward_timer;
  Timer backward_timer;
  Timer timer;
  std::vector<double> forward_time_per_layer(layers.size(), 0.0);
  std::vector<double> backward_time_per_layer(layers.size(), 0.0);
  double forward_time = 0.0;
  double backward_time = 0.0;
  for (int j = 0; j < FLAGS_iterations; ++j) {
    Timer iter_timer;
    iter_timer.Start();
    forward_timer.Start();
    for (int i = 0; i < layers.size(); ++i) {
      timer.Start();
      layers[i]->Forward(bottom_vecs[i], top_vecs[i]);
      forward_time_per_layer[i] += timer.MicroSeconds();
    }
    forward_time += forward_timer.MicroSeconds();
    backward_timer.Start();
    for (int i = layers.size() - 1; i >= 0; --i) {
      timer.Start();
      layers[i]->Backward(top_vecs[i], bottom_need_backward[i],
                          bottom_vecs[i]);
      backward_time_per_layer[i] += timer.MicroSeconds();
    }
    backward_time += backward_timer.MicroSeconds();
    LOG(INFO) << "Iteration: " << j + 1 << " forward-backward time: "
      << iter_timer.MilliSeconds() << " ms.";
  }
  LOG(INFO) << "Average time per layer: ";
  for (int i = 0; i < layers.size(); ++i) {
    const caffe::string& layername = layers[i]->layer_param().name();
    LOG(INFO) << std::setfill(' ') << std::setw(10) << layername <<
      "\tforward: " << forward_time_per_layer[i] / 1000 /
      FLAGS_iterations << " ms.";
    LOG(INFO) << std::setfill(' ') << std::setw(10) << layername  <<
      "\tbackward: " << backward_time_per_layer[i] / 1000 /
      FLAGS_iterations << " ms.";
  }
  total_timer.Stop();
  LOG(INFO) << "Average Forward pass: " << forward_time / 1000 /
    FLAGS_iterations << " ms.";
  LOG(INFO) << "Average Backward pass: " << backward_time / 1000 /
    FLAGS_iterations << " ms.";
  LOG(INFO) << "Average Forward-Backward: " << total_timer.MilliSeconds() /
    FLAGS_iterations << " ms.";
  LOG(INFO) << "Total Time: " << total_timer.MilliSeconds() << " ms.";
  LOG(INFO) << "*** Benchmark ends ***";
  return 0;
}
RegisterBrewFunction(time);

int main(int argc, char** argv) {
  // Print output to stderr (while still logging).
  // alsologtostderr define it in the gflags library
  FLAGS_alsologtostderr = 1;
  // Set version
  gflags::SetVersionString(AS_STRING(CAFFE_VERSION));
  // Usage message.
  gflags::SetUsageMessage("command line brew\n"
      "usage: caffe <command> <args>\n\n"
      "commands:\n"
      "  train           train or finetune a model\n"
      "  test            score a model\n"
      "  device_query    show GPU diagnostic information\n"
      "  time            benchmark model execution time");
  // Run tool or show usage.
  // initial flags
  caffe::GlobalInit(&argc, &argv);
  printf("argc=%d\n",argc);
  if (argc == 2) {
#ifdef WITH_PYTHON_LAYER
    try {
#endif
      return GetBrewFunction(caffe::string(argv[1]))();
#ifdef WITH_PYTHON_LAYER
    } catch (bp::error_already_set) {
      PyErr_Print();
      return 1;
    }
#endif
  } else {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/caffe");
  }
}
