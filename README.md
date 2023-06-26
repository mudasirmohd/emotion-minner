Emotion Predictor
==================

--------------------------------------------------------------------------------------------------------------
Prerequisite:
 Create virtual environment.
 Run below set of commands to create and start virtual environment with all the required packages.

`virtualenv venv`
`source venv/bin/activate`
`pip install -r requirements.txt`

-----------------------------------------------------------------------------------------------------------------

The Project consists of two systems:

Training Pipeline And Infer Rest App.

  Steps to annotate, train model and use it in inference:

  1) Run emotion_mining module to annotate the un labelled data.
  It consumes csv file, with text in `Sentences` column,
  It will generate two files `labelled.csv` and `unlabelled.csv`
  You can change the name of columns or files by editing main of `base.emotion_mining.py`

  2) Use labelled data from previous step to train svm classifier model using tf-iff vectors
  For training the model just run `svm_trainer.base_trainer.py` , change the names of in or
  out files if need arises.

  3) Infer label of un labelled data using trained svm model
  For this process just run main of `svm_train.infer.py` . It will consume un_labelled data generated in
  step 1 and model generated in step 2 and generate a file named `new_labelled.csv` .The file will contain
  labels  for all un_labelled data .

  4) Merge both labelled.csv and new_labelled.csv in single file say(merged.csv)

  5) Train RNN model on the merged.csv
  To train the model run main of `base_rnn_text_classifier.py`. You can change name of in/out files
  or column names by editing the main of the above python file.
  It will generate a rnn model in data/checkpoints. You can change the name/path of model directory.

  6) Start rest app  by running main of rest_app.app.py . It will use infer class of rnn to infer label
  of input facebook post.
  once the app is running go to: http://localhost:5522/get_prediction?post=YOUR_TEXT
  It will return the target label of text.
