%比较TSVM和S4VMbest，S4VMmin，S4VMcom，S4VM的泛化性能

addpath('libsvm-mat-2.89-3-box constraint');
labelNum=10;
trial=100;
balance='b';
C1=100;
C2=0.1;
datasetName={'austra','australian','breastw','clean1','diabetes','german','haberman','heart','house','house-votes','ionosphere','isolet','liverDisorders','optdigits','vehicle','wdbc','SSL,set=1','SSL,set=2','SSL,set=3','SSL,set=4','SSL,set=5','SSL,set=7','SSL,set=9'};
datasetNum=length(datasetName);

ttestRBF=zeros(datasetNum,5);
ttestLinear=zeros(datasetNum,5);

for i=1:datasetNum
    if(i<17)
        datasetType=['UCI','_',balance];
        trialNum=30;
    else
        datasetType=['SSL','_',balance];
        trialNum=12;
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
        model=svmtrain(label,labelInstance,C,'-t 0');
        predictLabel=svmpredict(groundTruth,testInstance,model);
        [resultLinearsvm(j,1),resultLinearsvm(j,2),resultLinearsvm(j,3)]=calculateAccAucF1(predictLabel,groundTruth);
        
        model=svmtrain(label,labelInstance,C,['-g ',num2str(gamma)]);
        predictLabel=svmpredict(groundTruth,testInstance,model);
        [resultRBFsvm(j,1),resultRBFsvm(j,2),resultRBFsvm(j,3)]=calculateAccAucF1(predictLabel,groundTruth);
        
%         % 计算TSVM的结果
%         predictLabel=segments(instance',[label;zeros(trainNum,1)],testInstance',optLinear);
%         [resultLineartsvm(j,1),resultLineartsvm(j,2),resultLineartsvm(j,3)]=calculateAccAucF1(predictLabel,groundTruth);
%         predictLabel=segments(instance',[label;zeros(trainNum,1)],testInstance',optRBF);
%         [resultRBFtsvm(j,1),resultRBFtsvm(j,2),resultRBFtsvm(j,3)]=calculateAccAucF1(predictLabel,groundTruth);
        
        C=[ones(labelNum,1)*C1;ones(trainNum,1)*C2];
        % 计算S4VM的结果
        start=cputime;
        [resultLinearbest(j,:),resultLinearminimum(j,:),resultLinearcom(j,:),resultLinears4vm(j,:)]=S4VM(instance,label,trainLabel,testInstance,groundTruth,labelNum,trainNum,0,C,trial,datasetName{i},j,'Linear');
        timeLinear(j)=cputime-start;
        start=cputime;
        [resultRBFbest(j,:),resultRBFminimum(j,:),resultRBFcom(j,:),resultRBFs4vm(j,:)]=S4VM(instance,label,trainLabel,testInstance,groundTruth,labelNum,trainNum,gamma,C,trial,datasetName{i},j,'RBF');
        timeRBF(j)=cputime-start;
        
    end
    
    % 评估结果
    ttestLinear(i,1)=myttest(resultLinearsvm(:,1),resultLineartsvm(:,1));
    ttestLinear(i,2)=myttest(resultLinearsvm(:,1),resultLinearbest(:,1));
    ttestLinear(i,3)=myttest(resultLinearsvm(:,1),resultLinearminimum(:,1));
    ttestLinear(i,4)=myttest(resultLinearsvm(:,1),resultLinearcom(:,1));
    ttestLinear(i,5)=myttest(resultLinearsvm(:,1),resultLinears4vm(:,1));
    
    ttestRBF(i,1)=myttest(resultRBFsvm(:,1),resultRBFtsvm(:,1));
    ttestRBF(i,2)=myttest(resultRBFsvm(:,1),resultRBFbest(:,1));
    ttestRBF(i,3)=myttest(resultRBFsvm(:,1),resultRBFminimum(:,1));
    ttestRBF(i,4)=myttest(resultRBFsvm(:,1),resultRBFcom(:,1));
    ttestRBF(i,5)=myttest(resultRBFsvm(:,1),resultRBFs4vm(:,1));
    
    save(['result\',datasetType,'_',datasetName{i},'_S4VM_',num2str(labelNum),'.mat'],'ttestRBF','ttestLinear','resultRBFsvm','resultRBFtsvm','resultRBFbest','resultRBFminimum','resultRBFcom','resultRBFs4vm','resultLinearsvm','resultLineartsvm','resultLinearbest','resultLinearminimum','resultLinearcom','resultLinears4vm','timeRBF','timeLinear');
end