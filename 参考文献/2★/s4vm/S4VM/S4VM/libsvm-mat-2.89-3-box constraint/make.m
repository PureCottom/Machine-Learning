% This make.m is used under Windows

% mex -O -c svm.cpp
% mex -O -c svm_model_matlab.c
% mex -O svmtrain.c svm.obj svm_model_matlab.obj
% mex -O svmpredict.c svm.obj svm_model_matlab.obj
% mex -O libsvmread.c
% mex -O libsvmwrite.c
% 

mex -O -c -largeArrayDims svm.cpp
mex -O -c -largeArrayDims svm_model_matlab.c
mex -O -largeArrayDims svmtrain.c svm.obj svm_model_matlab.obj
mex -O -largeArrayDims svmpredict.c svm.obj svm_model_matlab.obj
mex -O -largeArrayDims libsvmread.c
mex -O -largeArrayDims libsvmwrite.c


% mex -O -c  svm.cpp
% mex -O -c  svm_model_matlab.c
% mex -O  svmtrain.c svm.obj svm_model_matlab.obj
% mex -O  svmpredict.c svm.obj svm_model_matlab.obj
% mex -O  libsvmread.c
% mex -O  libsvmwrite.c