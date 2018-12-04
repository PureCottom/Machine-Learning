% This make.m is used under Windows

% mex -O  -c ..\blas\*.c -outdir ..\blas
% mex -O  -c ..\linear.cpp
% mex -O  -c ..\tron.cpp
% mex -O  -c linear_model_matlab.c -I..\
% mex -O  train.c -I..\ tron.obj linear.obj linear_model_matlab.obj ..\blas\*.obj
% mex -O  predict.c -I..\ tron.obj linear.obj linear_model_matlab.obj ..\blas\*.obj
% mex -O  libsvmread.c
% mex -O  libsvmwrite.c


mex -O -largeArrayDims -c ..\blas\*.c -outdir ..\blas
mex -O -largeArrayDims -c ..\linear.cpp
mex -O -largeArrayDims -c ..\tron.cpp
mex -O -largeArrayDims -c linear_model_matlab.c -I..\
mex -O -largeArrayDims train.c -I..\ tron.obj linear.obj linear_model_matlab.obj ..\blas\*.obj
mex -O -largeArrayDims predict.c -I..\ tron.obj linear.obj linear_model_matlab.obj ..\blas\*.obj
mex -O -largeArrayDims libsvmread.c
mex -O -largeArrayDims libsvmwrite.c