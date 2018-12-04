% In this experiment, we use balance split. That is, the labeled data
% share a similar imbalance ratio to that of unlabeled data.

% Two example data "wdbc" and "heart" are used. BTW: The running time of
% heart is much less than wdbc.

addpath('libsvm-mat-2.89-3-box constraint');
C1=100;
C2=0.1;
sampleTime=100; % number of low density separator

load('australian.mat');
load('australian,splits,labeled=10.mat');
%load('heart.mat');

gamma=length(pdist(X))/sum(pdist(X)); %gamma of RBF kernel
times = 30;
accLinear=zeros(times,1);
accRBF=zeros(times,1);

for j=1:times
    
    index=idxLabs(j,:);
    labelInstance=X(index,:);
    label=y(index,:);
    
    index=idxUnls(j,:);
    unlabelInstance=X(index,:);
    groundTruth=y(index,:);
    
    prediction=S4VM(labelInstance,label,unlabelInstance,'Linear',C1,C2,sampleTime,0);
    accLinear(j)=length(find(prediction==groundTruth))/length(groundTruth);
    
    prediction=S4VM(labelInstance,label,unlabelInstance,'RBF',C1,C2,sampleTime,gamma);
    accRBF(j)=length(find(prediction==groundTruth))/length(groundTruth);
end
disp(mean(accLinear))
disp(mean(accRBF))
