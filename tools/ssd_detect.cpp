// This is a demo code for using a SSD model to do detection.
// The code is modified from examples/cpp_classification/classification.cpp.
// Usage:
//    ssd_detect [FLAGS] model_file weights_file list_file
//
// where model_file is the .prototxt file defining the network architecture, and
// weights_file is the .caffemodel file containing the network parameters, and
// list_file contains a list of image files with the format as follows:
//    folder/img1.JPEG
//    folder/img2.JPEG
// list_file can also contain a list of video files with the format as follows:
//    folder/video1.mp4
//    folder/video2.mp4
//
#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include<time.h>

#define USE_OPENCV 1
#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)

class Detector {
 public:
  Detector(const string& model_file,
           const string& weights_file,
           const string& mean_file,
           const string& mean_value);

  std::vector<vector<float> > Detect(const cv::Mat& img);
  void showParams(void);

 private:
  void SetMean(const string& mean_file, const string& mean_value);

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);

 private:
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  cv::Mat mean_;
};

Detector::Detector(const string& model_file,
                   const string& weights_file,
                   const string& mean_file,
                   const string& mean_value) {
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
#else
  Caffe::set_mode(Caffe::GPU);
#endif

  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(weights_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

  /* Load the binaryproto mean file. */
  SetMean(mean_file, mean_value);
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
              //dataTmp =outPutFeature[i*featureBlob->height()*featureBlob->width()+j];
              dataTmp =outPutFeature[i+j*featureBlob->channels()];
              if(maxData < dataTmp){
                  maxData = dataTmp;
              }
              if(minData > dataTmp){
                  minData = dataTmp;
              }
              if(  (0.00001f > dataTmp) && (-0.00001f < dataTmp)){ /// only print 0.00f,no -0.000f
                  dataTmp = 0 ; }
              fprintf(oFile,"%4f\n",float(dataTmp));
          }
       }
        printf("minData,maxData=%f,%f\n",minData,maxData);
        fclose(oFile);
     }
}

cv::Mat CombineMultiImages(const std::vector<cv::Mat>& Images,
                           const int NumberOfRows,
                           const int NumberOfCols,
                           const int Distance,
                           const int ImageWidth,
                           const int ImageHeight)
{
    // return empty mat if the Number of rows or cols is smaller than 1.
    assert((NumberOfRows > 0) && (NumberOfCols > 0));
    if ((NumberOfRows < 1) || (NumberOfCols < 1))
    {
        std::cout << "The number of the rows or the cols is smaller than 1."
                  << std::endl;
        return cv::Mat();
    }


    // return empty mat if the distance, the width or the height of image
    // is smaller than 1.
    assert((Distance > 0) && (ImageWidth > 0) && (ImageHeight > 0));
    if ((Distance < 1) || (ImageWidth < 1) || (ImageHeight < 1))
    {
        std::cout << "The distance, the width or the height of the image is smaller than 1."
                  << std::endl;
        return cv::Mat();
    }


    // Get the number of the input images
    const int NUMBEROFINPUTIMAGES = Images.size();


    // return empty mat if the number of the input images is too big.
    assert(NUMBEROFINPUTIMAGES <= NumberOfRows * NumberOfCols);
    if (NUMBEROFINPUTIMAGES > NumberOfRows * NumberOfCols)
    {
        std::cout << "The number of images is too big." << std::endl;
        return cv::Mat();
    }


    // return empty mat if the number of the input images is too low.
    assert(NUMBEROFINPUTIMAGES > 0);
    if (NUMBEROFINPUTIMAGES < 1)
    {
        std::cout << "The number of images is too low." << std::endl;
        return cv::Mat();
    }


    // create the big image
    const int WIDTH = Distance * (NumberOfCols + 1) + ImageWidth * NumberOfCols;
    const int HEIGHT = Distance * (NumberOfRows + 1) + ImageHeight * NumberOfRows;
    cv::Scalar Color(255, 255, 255);
    if (Images[0].channels() == 1)
    {
        Color = cv::Scalar(255);
    }
    cv::Mat ResultImage(HEIGHT, WIDTH, Images[0].type(), Color);



    // copy the input images to the big image
    for (int Index = 0; Index < NUMBEROFINPUTIMAGES; Index++)
    {

        assert(Images[Index].type() == ResultImage.type());
        if (Images[Index].type() != ResultImage.type())
        {
            std::cout << "The No." << Index << "image has the different type."
                      << std::endl;
            return cv::Mat();
        }


        // Get the row and the col of No.Index image
        int Rows = Index / NumberOfCols;
        int Cols = Index % NumberOfCols;

        // The start point of No.Index image.
        int StartRows = Distance * (Rows + 1) + ImageHeight * Rows;
        int StartCols = Distance * (Cols + 1) + ImageWidth * Cols;

        // copy  No.Index image to the big image
        cv::Mat ROI = ResultImage(cv::Rect(StartCols, StartRows,
                                           ImageWidth, ImageHeight));
        cv::resize(Images[Index], ROI, cv::Size(ImageWidth, ImageHeight));

    }

    return ResultImage;
}


void Detector::showParams(void)
{

	vector<shared_ptr<Blob<float> > > params = net_->params();
//
//	for (int i = 0; i < params.size(); i++)
//	{
//
//		std::cout<<params[i].get()->shape_string()<<std::endl;
//	}

	Blob<float>* allParamsData = params[0].get();
	const float* paramsData = allParamsData->cpu_data();

	for (int i = 0; i < 64*3*3*3; i++)
	{
//		if (paramsData[i] >= 1.0)
//		{
//			std::cout<<paramsData[i]<<" ";
//		}
		std::cout<<paramsData[i]<<" ";

	}

}


std::vector<vector<float> > Detector::Detect(const cv::Mat& img) {
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  Preprocess(img, &input_channels);

  system("rm -rf cnnData");
  system("cp -rf emptyCnn cnnData");
  net_->Forward();
  logCaffeBlob(*net_);
//  showParams();
#if 0
  vector<string> blobs_names = net_->blob_names();
  vector<shared_ptr<Blob<float> > > blobs = net_->blobs();
  for(int i = 0; i < blobs_names.size(); i++)
  {
	  std::cout<<blobs_names[i];
	  vector<int> shape = blobs[i].get()->shape();
	  CoutBlobShape(shape);
  }

  ////////==== debugPrint Begin
//  string strName="pool6";
//  char * strTmp="Pool6_";

//  string strName="detection_out";
//  char * strTmp="Detection_Out";

//    string strName="fc7";
//    char * strTmp="fc7Out";

//      string strName="conv4_3_norm";
//      char * strTmp="conv4_3_normOut";


  std::cout<<strName;
//  std::cout<<net_->getBlobByName(strName)->shape_string()<<std::endl;
  //CoutBlobShape(shape);


	static int printConvolutionLayerLog = 0 ;
	printConvolutionLayerLog = printConvolutionLayerLog + 1;
	char datFileReluName[128]={0};
	std::ofstream oFileData;
	sprintf(datFileReluName,"cnnData/stimulusCNNBlob%s%d.dat",strTmp,printConvolutionLayerLog);
//	strName.append(datFileReluName);
	oFileData.open(datFileReluName);
//	oFileData.open(strName);



  Blob<float>* featureBlob = net_->getBlobByName(strName);
  const float* outPutFeature = featureBlob->cpu_data();

  void *dstPtr = (void *)malloc(60*60*4);

  std::cout<<featureBlob->count()<<std::endl;

  std::vector<cv::Mat> Images;
  std::cout<<"featureBlob->channels()="<<featureBlob->channels()<<std::endl;
  std::cout<<"featureBlob->height()="<<featureBlob->height()<<std::endl;
  std::cout<<"featureBlob->width()="<<featureBlob->width()<<std::endl;
  for (int i = 0; i < featureBlob->channels(); i++)
  {
//	  outPutFeature += 1 * featureBlob->height() * featureBlob->width();

	  for (int j=0; j < featureBlob->height() * featureBlob->width(); j++)
	  {
			 oFileData<<outPutFeature[i*featureBlob->height() * featureBlob->width()+j]<<std::endl;

	  }
 }

  oFileData.close();
////////==== debugPrint End

//	 cv::Mat img(featureBlob->height(),featureBlob->width(),CV_32F, (void*)outPutFeature);

//	 cv::Mat dst(60, 60, CV_32F, dstPtr);
//	 cv::resize(img, dst, cv::Size(60, 60));
//	 Images.push_back(img);
//	 if (i == 35)
//	 {
//		 break;
//	 }
////
//
//	 ostringstream stream;
//	 stream << "feature" << i+1;
//
//		 cv::imshow(stream.str(), img);




//  cv::Mat features = CombineMultiImages(Images, 8, 8, 10, 60, 60);

//  for (int i=0; i < featureBlob->channels(); i++)
//  {
//	  	 ostringstream stream;
//	  	 stream << "feature" << i+1;
//	  	 if(i==15)
//	  	 {
//	  		 cv::imshow("feature", Images[i]);
//	  		cv::waitKey(0);
//	  	 }
//  }
//  cv::imshow("features", features);
//  cv::waitKey(0);
#endif
  /* Copy the output layer to a std::vector */
  Blob<float>* result_blob = net_->output_blobs()[0];
  const float* result = result_blob->cpu_data();
  const int num_det = result_blob->height();
  vector<vector<float> > detections;
  for (int k = 0; k < num_det; ++k) {
    if (result[0] == -1) {
      // Skip invalid detection.
      result += 7;
      continue;
    }
    vector<float> detection(result, result + 7);
    detections.push_back(detection);
    result += 7;
  }
  return detections;
}



/* Load the mean file in binaryproto format. */
void Detector::SetMean(const string& mean_file, const string& mean_value) {
  cv::Scalar channel_mean;
  if (!mean_file.empty()) {
    CHECK(mean_value.empty()) <<
      "Cannot specify mean_file and mean_value at the same time";
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

    /* Convert from BlobProto to Blob<float> */
    Blob<float> mean_blob;
    mean_blob.FromProto(blob_proto);
    CHECK_EQ(mean_blob.channels(), num_channels_)
      << "Number of channels of mean file doesn't match input layer.";

    /* The format of the mean file is planar 32-bit float BGR or grayscale. */
    std::vector<cv::Mat> channels;
    float* data = mean_blob.mutable_cpu_data();
    for (int i = 0; i < num_channels_; ++i) {
      /* Extract an individual channel. */
      cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
      channels.push_back(channel);
      data += mean_blob.height() * mean_blob.width();
    }

    /* Merge the separate channels into a single image. */
    cv::Mat mean;
    cv::merge(channels, mean);

    /* Compute the global mean pixel value and create a mean image
     * filled with this value. */
    channel_mean = cv::mean(mean);
    mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
  }
  if (!mean_value.empty()) {
    CHECK(mean_file.empty()) <<
      "Cannot specify mean_file and mean_value at the same time";
    stringstream ss(mean_value);
    vector<float> values;
    string item;
    while (getline(ss, item, ',')) {
      float value = std::atof(item.c_str());
      values.push_back(value);
    }
    CHECK(values.size() == 1 || values.size() == num_channels_) <<
      "Specify either 1 mean_value or as many as channels: " << num_channels_;

    std::vector<cv::Mat> channels;
    for (int i = 0; i < num_channels_; ++i) {
      /* Extract an individual channel. */
      cv::Mat channel(input_geometry_.height, input_geometry_.width, CV_32FC1,
          cv::Scalar(values[i]));
      channels.push_back(channel);
    }
    cv::merge(channels, mean_);
  }
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Detector::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void Detector::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  cv::Mat sample_normalized;
  cv::subtract(sample_float, mean_, sample_normalized);

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_normalized, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

DEFINE_string(mean_file, "",
    "The mean file used to subtract from the input image.");
DEFINE_string(mean_value, "104,117,123",
    "If specified, can be one value or can be same as image channels"
    " - would subtract from the corresponding channel). Separated by ','."
    "Either mean_file or mean_value should be provided, not both.");
DEFINE_string(file_type, "image",
    "The file type in the list_file. Currently support image and video.");
DEFINE_string(out_file, "",
    "If provided, store the detection results in the out_file.");
DEFINE_double(confidence_threshold, 0.01,
    "Only store detections with score higher than the threshold.");

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Do detection using SSD mode.\n"
        "Usage:\n"
        "    ssd_detect [FLAGS] model_file weights_file list_file\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 4) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "examples/ssd/ssd_detect");
    return 1;
  }

  const string& model_file = argv[1];
  const string& weights_file = argv[2];
  const string& mean_file = FLAGS_mean_file;
  const string& mean_value = FLAGS_mean_value;
  const string& file_type = FLAGS_file_type;
  const string& out_file = FLAGS_out_file;
  const float confidence_threshold = FLAGS_confidence_threshold;

  // Initialize the network.
  Detector detector(model_file, weights_file, mean_file, mean_value);

  // Set the output mode.
  std::streambuf* buf = std::cout.rdbuf();
  std::ofstream outfile;
  if (!out_file.empty()) {
    outfile.open(out_file.c_str());
    if (outfile.good()) {
      buf = outfile.rdbuf();
    }
  }
  std::ostream out(buf);

  // Process image one by one.
  std::ifstream infile(argv[3]);
  std::string file;

  struct timeval start, end;

  while (infile >> file) {
    if (file_type == "image") {
      cv::Mat img = cv::imread(file, -1);
      CHECK(!img.empty()) << "Unable to decode image " << file;

      gettimeofday(&start, NULL);

      std::vector<vector<float> > detections = detector.Detect(img);

      gettimeofday(&end, NULL);
      std::cout<<"detection time: "<<(end.tv_sec-start.tv_sec) + 0.000001*float((end.tv_usec-start.tv_usec))<<std::endl;

      /* Print the detection results. */
      for (int i = 0; i < detections.size(); ++i) {
        const vector<float>& d = detections[i];
        // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
        CHECK_EQ(d.size(), 7);
        const float score = d[2];
        if (score >= confidence_threshold) {
//        if (score >= 0.2) {
          out << file << " ";
          out << static_cast<int>(d[1]) << " ";
          out << score << " ";
          out << static_cast<int>(d[3] * img.cols) << " ";
          out << static_cast<int>(d[4] * img.rows) << " ";
          out << static_cast<int>(d[5] * img.cols) << " ";
          out << static_cast<int>(d[6] * img.rows) << std::endl;
          cv::rectangle(img,cvPoint(d[3] * img.cols,d[4] * img.rows),
              		  cvPoint(d[5] * img.cols,d[6] * img.rows),cvScalar(0,0,255),2);
        }
      }

      cv::imshow("detection", img);
      cv::waitKey(0);
    } else if (file_type == "video") {
      cv::VideoCapture cap(file);
      if (!cap.isOpened()) {
        LOG(FATAL) << "Failed to open video: " << file;
      }
      cv::Mat img;
      int frame_count = 0;
      while (true) {
        bool success = cap.read(img);
        if (!success) {
          LOG(INFO) << "Process " << frame_count << " frames from " << file;
          break;
        }
        CHECK(!img.empty()) << "Error when read frame";
        std::vector<vector<float> > detections = detector.Detect(img);

        /* Print the detection results. */
        for (int i = 0; i < detections.size(); ++i) {
          const vector<float>& d = detections[i];
          // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
          CHECK_EQ(d.size(), 7);
          const float score = d[2];
          if (score >= confidence_threshold) {
            out << file << "_";
            out << std::setfill('0') << std::setw(6) << frame_count << " ";
            out << static_cast<int>(d[1]) << " ";
            out << score << " ";
            out << static_cast<int>(d[3] * img.cols) << " ";
            out << static_cast<int>(d[4] * img.rows) << " ";
            out << static_cast<int>(d[5] * img.cols) << " ";
            out << static_cast<int>(d[6] * img.rows) << std::endl;
          }
        }
        ++frame_count;
      }
      if (cap.isOpened()) {
        cap.release();
      }
    } else {
      LOG(FATAL) << "Unknown file_type: " << file_type;
    }
  }
  return 0;
}
#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV
