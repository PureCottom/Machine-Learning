%比较TSVM和S4VMbest，S4VMmin，S4VMcom，S4VM的泛化性能

addpath('libsvm-mat-2.89-3-box constraint');
labelNum=10;
trial=100;
balance='b';
C1=100;
C2=0.1;
datasetName={'austra','australian','breastw','clean1','diabetes','haberman','heart','house','house-votes','ionosphere','isolet','liverDisorders','optdigits','vehicle','wdbc','SSL,set=1','SSL,set=2','SSL,set=3','SSL,set=4','SSL,set=5'};
datasetNum=length(datasetName);

ttestRBF=zeros(datasetNum,5);
ttestLinear=zeros(datasetNum,5);

for i=1:datasetNum
    if(i<16)
        datasetType=['UCI','_',balance];
        trialNum=2;
    else
        datasetType=['SSL','_',balance];
        trialNum=2;
    end
    
    disp(i);
    load(['dataset/',datasetType,'/',datasetName{i},'.mat']);
    load(['dataset/',datasetType,'/',datasetName{i},',splits,labeled=',num2str(labelNum),'.mat']);
    X=normalization(X);
    
    optLinear=struct('t',0,'c',100);
    resultLinearsvm=zeros(trialNum,3);
    resultLineartsvm=zeros(trialNum,3);
    resultLinearbest=zeros(trialNum,3);
    resultLinearminimum=zeros(trialNum,3);
    resultLinearcom=zeros(trialNum,3);
    resultLinears4vm=zeros(trialNum,3);
    
    gamma=length(pdist(X))/sum(pdist(X));
    optRBF=struct('t',2,'c',100,'g',gamma);  
    resultRBFsvm=zeros(trialNum,3);
    resultRBFtsvm=zeros(trialNum,3);
    resultRBFbest=zeros(trialNum,3);
    resultRBFminimum=zeros(trialNum,3);
    resultRBFcom=zeros(trialNum,3);
    resultRBFs4vm=zeros(trialNum,3);
    
    
    timeRBF=zeros(trialNum,1);
    timeLinear=zeros(trialNum,1);
    
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
        % 计算SVM的结果
%         model=svmtrain(label,labelInstance,C,'-t 0');
%         predictLabel=svmpredict(groundTruth,testInstance,model);
%         [resultLinearsvm(j,1),resultLinearsvm(j,2),resultLinearsvm(j,3)]=calculateAccAucF1(predictLabel,groundTruth);
        
%         model=svmtrain(label,labelInstance,C,['-g ',num2str(gamma)]);
%         predictLabel=svmpredict(groundTruth,testInstance,model);
%         [resultRBFsvm(j,1),resultRBFsvm(j,2),resultRBFsvm(j,3)]=calculateAccAucF1(predictLabel,groundTruth);
        
%         % 计算TSVM的结果
        [LinearS3VM_train_time(j),LinearS3VM_test_time(j)] =segments_cputime(instance',[label;zeros(trainNum,1)],testInstance',optLinear);
%         [resultLineartsvm(j,1),resultLineartsvm(j,2),resultLineartsvm(j,3)]=calculateAccAucF1(predictLabel,groundTruth);
        [RBFS3VM_train_time(j),RBFS3VM_test_time(j)] = segments_cputime(instance',[label;zeros(trainNum,1)],testInstance',optRBF);
%         [resultRBFtsvm(j,1),resultRBFtsvm(j,2),resultRBFtsvm(j,3)]=calculateAccAucF1(predictLabel,groundTruth);
        
        C=[ones(labelNum,1)*C1;ones(trainNum,1)*C2];
        % 计算S4VM的结果

        [LinearS4VM_train_time(j),LinearS4VM_test_time(j)] = S4VM_cputime(instance,label,trainLabel,testInstance,groundTruth,labelNum,trainNum,0,C,trial,datasetName{i},j,'Linear');

        
        [RBFS4VM_train_time(j),RBFS4VM_test_time(j)] = S4VM_cputime(instance,label,trainLabel,testInstance,groundTruth,labelNum,trainNum,gamma,C,trial,datasetName{i},j,'RBF');

    end
    
    
    save(['cpu_time\',datasetType,'_',datasetName{i},'_S4VM_',num2str(labelNum),'.mat'],...
        'LinearS3VM_train_time','LinearS3VM_test_time','RBFS3VM_train_time','RBFS3VM_test_time',...
        'LinearS4VM_train_time','LinearS4VM_test_time','RBFS4VM_train_time','RBFS4VM_test_time');
end