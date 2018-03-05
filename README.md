# Live_FaceRecognition

Overview

learningrecognizer ---> package where all learning Algo related things will be there

  - preprocessing ---> Detection and getting encodings data file
          -datafiles ---> Encodings File got generated after preprocessing
          -orl_faces ---> Training and Testing images
          -FaceDetector.py ---> Starting point for the detection. It does face_detection, pose_Prediction and 
                                calls the extraction method for encodings.
          -ImageClass ---> Used to encapsulate all the images atuff at one place
  
  - encodings.py ---> Extracts encodings for each face detected
  
  - learningSVM.py ---> learns SVM from the generated encodings
  
​

Order of Execution

1)FaceDetector.py - Creates the data file(ListOfEncodings.csv)

2)learningSVM.py - Uses Data file for Training SVM
