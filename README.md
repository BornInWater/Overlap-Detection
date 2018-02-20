
# Overlapped Speech detection in multi-party conversations

### Leveraging LSTM models for overlap detection in multi-party meetings 
    Neeraj Sajjan∗, Shobhana Ganesh∗, Neeraj Sharma∗, Sriram Ganapathy∗, Neville Ryant+
    
    ∗Learning and Extraction of Acoustic Patterns (LEAP) Lab, Indian Institute of Science, Bangalore-560012
    +Linguistic Data Consortium, University of Pennsylvania, USA



This is a project on overlapped speech detection in multi-party conversation meetings. We explore the effectivness of various features such as Mel Spectrogram, kurtosis etc using a neural network approach on two datasets: [TIMIT](https://catalog.ldc.upenn.edu/ldc93s1) and [AMI](http://groups.inf.ed.ac.uk/ami/corpus/). We make use of force alignment to rectify the errors inherent in human annotations of the AMI dataset.

## Dependencies
1. Python 2.7 
2. [HTK Toolkit](http://htk.eng.cam.ac.uk/)
3. [Keras](https://keras.io/)

## Folder Structure and Description
### Codes
1. *Feature Handling Codes*:
  - htkmfc.py : python interface to reading and writing htk files
  - htk_dataprep.sh : shell script to generate htk feature files
  - mfcc_config.cfg : config file for mel spectorgram[fbank] feature generation
  - get_gammatone_feats.py : code for extraction of gammatone spectorgram features
  - gammatonegram_package.py : necessary functions for get_gammatone_feats.py
  - generator.py : code to generate artifical overlapped speech wav file from two single speaker files[for TIMIT]
  - wrapper_for_gen.py : wrapper for generator.py, takes in a list of single speaker wav files to generate overlapped wav files
  - PAR_make_context_feats.py : Code to generate context features for multiple files parallely
  - PAR_do_cmvn_feats.py : Code for cepstral mean variance normalisation
  - concatenator.py : Final train/val/test data generation code using htk file format
  - kurtosis_extractor.py : Code to extract kurtosis
  - sfm.py : Code to extract spectral flatness measure features
  
2. *Model_Train_Test_Codes*:
  - rnn.py : Code for final lstm model in Keras
  - cnn.py : CNN model
  - dnn.py : Three layered dnn
  - clstm.py : CNN followed by LSTM
  - test_rnn.py : Code for testing final lstm model
  - confusion_matrix_gen.py : Code to generate confusion matrix for three classes[Single/Overlap/Filler]
 
 ### Labels
 1. *Original_Labels* : Labels of AMI train, dev[val], eval[test] sets before force alignment.
 2. *Force_Aligned_Labels* : After Force alignment


