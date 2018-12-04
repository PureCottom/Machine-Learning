import weka.classifiers.trees.*;
import weka.core.*;
import java.io.*;
import java.lang.*;

datasetName={'artificial','auto-mpg','flag','hayes-roth','hypothyroid','iris','letter','machine','page-blocks','segment','vehicle','wine'};
data_ext='.arff';

a=load('costMatrix\costMatrix.mat');
a=a.CostMatrix;

TRAIL=5;%5 times
CVTIMES=10;%10-fold

for i=7:7
    data_file=String(['dataset/',datasetName{i},data_ext]);
    r = FileReader(data_file);
    data=Instances(r);
    data.setClassIndex(data.numAttributes()-1);
    clear r datafile;

    n=data.numClasses(); %number of class label

    CostMatrix=a{i};
    
    %one_vs_all(datasetName{i},data,n,CostMatrix,TRAIL,CVTIMES);
    one_vs_one(datasetName{i},data,n,CostMatrix,TRAIL,CVTIMES);
    %closed_form_reduction(datasetName{i},data,n,CostMatrix,TRAIL,CVTIMES);
end