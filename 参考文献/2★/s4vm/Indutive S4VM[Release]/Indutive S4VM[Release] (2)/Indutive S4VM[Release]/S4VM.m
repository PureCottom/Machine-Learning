function [best,minimum,com,s4vm]=S4VM(instance,label,trainLabel,testInstance,groundTruth,labelNum,unlabelNum,gamma,C,trial)

best=zeros(1,3);
minimum=zeros(1,3);
com=zeros(1,3);
s4vm=zeros(1,3);

beta=sum(label)/length(label);
alpha=0.1;
clusterNum=floor(trial/10);%the number of cluster
Y=zeros(trial+1,labelNum+unlabelNum);
S=zeros(trial+1,2);
V=zeros(trial+1,labelNum+unlabelNum);
M=cell(trial+1,1);

%localDescent
if(gamma==0)
    model=svmtrain(label,instance(1:labelNum,:),C(1:labelNum),'-t 0');
else
    model=svmtrain(label,instance(1:labelNum,:),C(1:labelNum),['-g ',num2str(gamma)]);
end
[ysvm,~,~]=svmpredict([label;trainLabel],instance,model);
[ysvmpre,~,~]=svmpredict(groundTruth,testInstance,model);

%check if trival solution
if(sum(ysvm(labelNum+1:labelNum+unlabelNum)>0)==0||sum(ysvm(labelNum+1:labelNum+unlabelNum)<0)==0)
    Y=Y(1:trial,:);
    S=S(1:trial,:);
    M=M(1:trial);
else
    [predictBest,~,valuesBest,modelBest]=localDescent1(instance,ysvm,labelNum,unlabelNum,gamma,C,beta,alpha);
    Y(trial+1,:)=predictBest;
    S(trial+1,1)=sum(predictBest(labelNum+1:labelNum+unlabelNum)==trainLabel)/length(trainLabel);
    S(trial+1,2)=modelBest.obj;
    V(trial+1,:)=valuesBest;
    M{trial+1}=modelBest;
end

for i=1:trial
    
    disp(i);
    %generate the label for unlabeled data randomly
    if(i<=trial*0.8)
        y=rand(unlabelNum,1);
        y(y>0.5)=1;
        y(y<=0.5)=-1;
        labelNew=[label;y];
    else
        y=rand(unlabelNum,1);
        y(y>0.8)=-1;
        y(y<=0.8)=1;
        labelNew=[label;y.*ysvm(labelNum+1:labelNum+unlabelNum)];
    end
    [predictBest,~,valuesBest,modelBest]=localDescent1(instance,labelNew,labelNum,unlabelNum,gamma,C,beta,alpha);
    
    %record the prediction
    Y(i,:)=predictBest;
    
    %record the accuracy and objective
    S(i,1)=sum(predictBest(labelNum+1:labelNum+unlabelNum)==trainLabel)/length(trainLabel);
    S(i,2)=modelBest.obj;
    V(i,:)=valuesBest;
    M{i}=modelBest;
end

%Use K-means algorithm to cluster the predictions
[IDX,~,~,D]=kmeans(Y,clusterNum,'Distance','cityblock','EmptyAction','drop');

%exclude the empty cluster
D=sum(D,1);
clusterIndex=find(isnan(D)==0);
clusterNum=size(find(isnan(D)==0),2);

%record the best objective in each cluster
obj=zeros(clusterNum,1);

%Record all the predictions. One for each cluster
prediction=zeros(labelNum+unlabelNum,clusterNum);
%Record all the predictive values. One for each cluster
values=zeros(labelNum+unlabelNum,clusterNum);
%Record the accuracy for each candidate
accuracy=zeros(clusterNum,1);
model=cell(clusterNum,1);

for i=1:clusterNum
    %find the indexes corresponding to each cluster
    index=find(IDX==clusterIndex(i));
    
    %extract the information for each cluster
    tempS=S(index,:);
    tempY=Y(index,:);
    tempV=V(index,:);
    tempM=M(index);
    
    %record the objectives in each cluster
    obj2=tempS(:,2);
    
    %select the best one and record
    [~,index2]=max(obj2);
    accuracy(i)=tempS(index2,1); 
    values(:,i)=tempV(index2,:); 
    obj(i)=tempS(index2,2);
    prediction(:,i)=tempY(index2,:)';
    model{i}=tempM(index2); 
end

%record the best performance for the candidates in terms of accuracy, AUC
%and F1
[~,index]=max(accuracy);
predictLabel=svmpredict(groundTruth,testInstance,model{index}{1});
[best(1),best(2),best(3)]=calculateAccAucF1(predictLabel,groundTruth);
        
%record the performance of the best objective in terms of accuracy, AUC
%and F1
[~,index]=max(obj);
predictLabel=svmpredict(groundTruth,testInstance,model{index}{1});
[minimum(1),minimum(2),minimum(3)]=calculateAccAucF1(predictLabel,groundTruth);

%record the voting results of the candidates
predictLabel=zeros(length(groundTruth),1);
for i=1:clusterNum
    predictLabel=predictLabel+svmpredict(groundTruth,testInstance,model{i}{1});
end
predictLabel=sign(predictLabel);
[com(1),com(2),com(3)]=calculateAccAucF1(predictLabel,groundTruth);

%the result of s4vm
testNum=length(groundTruth);
pre=zeros(testNum,clusterNum);
for i=1:clusterNum
    pre(:,i)=svmpredict(groundTruth,testInstance,model{i}{1});
end
label=groundTruth;
for i=1:length(groundTruth)
    temp=linearProgramming([prediction;pre(i,:)],[ysvm;ysvmpre(i)],labelNum,3);
    label(i)=temp(length(temp));
end
[s4vm(1),s4vm(2),s4vm(3)]=calculateAccAucF1(label,groundTruth);

end
