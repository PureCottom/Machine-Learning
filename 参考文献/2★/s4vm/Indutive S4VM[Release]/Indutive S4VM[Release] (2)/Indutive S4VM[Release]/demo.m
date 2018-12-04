function demo()

addpath('libsvm-mat-2.89-3-box constraint');
labelNum=10;
trialNum=10;
C1=100;
C2=0.1;
trial=100;
datasetName='breastw';
datasetNum=length(datasetName);

load([datasetName,'.mat']);
load([datasetName,',splits,labeled=',num2str(labelNum),'.mat']);
X=normalization(X);

optLinear=struct('t',0,'c',100);
resultLinearsvm=zeros(trialNum,3);
resultLinears3vm = zeros(trialNum,3);
resultLineartsvm=zeros(trialNum,3);
resultLinearbest=zeros(trialNum,3);
resultLinearminimum=zeros(trialNum,3);
resultLinearcom=zeros(trialNum,3);
resultLinears4vm=zeros(trialNum,3);

gamma=length(pdist(X))/sum(pdist(X));
optRBF=struct('t',2,'c',100,'g',gamma);  
resultRBFsvm=zeros(trialNum,3);
resultRBFs3vm=zeros(trialNum,3);
resultRBFtsvm=zeros(trialNum,3);
resultRBFbest=zeros(trialNum,3);
resultRBFminimum=zeros(trialNum,3);
resultRBFcom=zeros(trialNum,3);
resultRBFs4vm=zeros(trialNum,3);
    
    
for j=1:trialNum
    disp(j);
    index=idxLabs(j,:);
    labelInstance=X(index,:);
    label=y(index,:);
    index=idxUnls(j,:);
    trainInstance=X(index(1:trainNum),:);
    trainLabel=y(index(1:trainNum));
    testInstance=X(index(trainNum+1:trainNum+testNum),:);
    groundTruth=y(index(trainNum+1:trainNum+testNum));

    instance=[labelInstance;trainInstance];

    C=ones(labelNum,1)*C1;
    
    % conduct the result of supervised SVM with only labeled data
    model=svmtrain(label,labelInstance,C,'-t 0');
    predictLabel=svmpredict(groundTruth,testInstance,model);
    [resultLinearsvm(j,1),resultLinearsvm(j,2),resultLinearsvm(j,3)]=calculateAccAucF1(predictLabel,groundTruth);

    model=svmtrain(label,labelInstance,C,['-g ',num2str(gamma)]);
    predictLabel=svmpredict(groundTruth,testInstance,model);
    [resultRBFsvm(j,1),resultRBFsvm(j,2),resultRBFsvm(j,3)]=calculateAccAucF1(predictLabel,groundTruth);

    % conduct the result of S3VMs
    C=ones(labelNum,1)*C1;
    model=svmtrain(label,labelInstance,C,'-t 0');
    predictLabel=svmpredict(trainLabel,trainInstance,model);
    C=ones(labelNum+trainNum,1)*C1;
    model=svmtrain([label;predictLabel],instance,C,'-t 0');
    predictLabel=svmpredict(groundTruth,testInstance,model);
    [resultLinears3vm(j,1),resultLinears3vm(j,2),resultLinears3vm(j,3)]=calculateAccAucF1(predictLabel,groundTruth);

    C=ones(labelNum,1)*C1;
    model=svmtrain(label,labelInstance,C,['-g ',num2str(gamma)]);
    predictLabel=svmpredict(trainLabel,trainInstance,model);
    C=ones(labelNum+trainNum,1)*C1;
    model=svmtrain([label;predictLabel],instance,C,['-g ',num2str(gamma)]);
    predictLabel=svmpredict(groundTruth,testInstance,model);
    [resultRBFs3vm(j,1),resultRBFs3vm(j,2),resultRBFs3vm(j,3)]=calculateAccAucF1(predictLabel,groundTruth);



    % conduct the result of TSVM
    predictLabel=segments(instance',[label;zeros(trainNum,1)],testInstance',optLinear);
    [resultLineartsvm(j,1),resultLineartsvm(j,2),resultLineartsvm(j,3)]=calculateAccAucF1(predictLabel,groundTruth);
    predictLabel=segments(instance',[label;zeros(trainNum,1)],testInstance',optRBF);
    [resultRBFtsvm(j,1),resultRBFtsvm(j,2),resultRBFtsvm(j,3)]=calculateAccAucF1(predictLabel,groundTruth);


    % conduct the reuslt of S4VM
    C=[ones(labelNum,1)*C1;ones(trainNum,1)*C2];
    start=cputime;
    [resultLinearbest(j,:),resultLinearminimum(j,:),resultLinearcom(j,:),resultLinears4vm(j,:)]=S4VM(instance,label,trainLabel,testInstance,groundTruth,labelNum,trainNum,0,C,trial);
    timeLinear(j)=cputime-start;
    start=cputime;
    [resultRBFbest(j,:),resultRBFminimum(j,:),resultRBFcom(j,:),resultRBFs4vm(j,:)]=S4VM(instance,label,trainLabel,testInstance,groundTruth,labelNum,trainNum,gamma,C,trial);
    timeRBF(j)=cputime-start;

end
    

save([datasetName,'_S3VM_',num2str(labelNum),'.mat'],...
    'ttestRBF','ttestLinear','resultRBFsvm','resultRBFs3vm',...
    'resultLinearsvm','resultLinears3vm',...
    'resultLineartsvm','resultRBFtsvm',...
    'resultLinearbest','resultLinearminimum','resultLinearcom','resultLinears4vm',...
    'resultRBFbest','resultRBFminimum','resultRBFcom','resultRBFs4vm');
