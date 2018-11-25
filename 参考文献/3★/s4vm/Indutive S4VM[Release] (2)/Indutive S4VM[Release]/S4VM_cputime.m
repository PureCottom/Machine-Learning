function [train_time,test_time] = S4VM_cputime(instance,label,trainLabel,testInstance,groundTruth,labelNum,unlabelNum,gamma,C,trial,datasetName,trialNum,kernel)


start = cputime;

% best=zeros(1,3);
% minimum=zeros(1,3);
% com=zeros(1,3);
% s4vm=zeros(1,3);

beta=sum(label)/length(label);
alpha=0.1;
clusterNum=floor(trial/10);%聚类的个数
Y=zeros(trial+1,labelNum+unlabelNum);
S=zeros(trial+1,2);
V=zeros(trial+1,labelNum+unlabelNum);
M=cell(trial+1,1);

%使用SVM对未标记样本的预测作为初始下降点做一次localDescent
if(gamma==0)
    model=svmtrain(label,instance(1:labelNum,:),C(1:labelNum),'-t 0');
else
    model=svmtrain(label,instance(1:labelNum,:),C(1:labelNum),['-g ',num2str(gamma)]);
end
[ysvm,tmp1,tmp2]=svmpredict([label;trainLabel],instance,model);
[ysvmpre,tmp1,tmp2]=svmpredict(groundTruth,testInstance,model);

%如果SVM将所有未标记样本都归为一类，就放弃，否则继续做
if(sum(ysvm(labelNum+1:labelNum+unlabelNum)>0)==0||sum(ysvm(labelNum+1:labelNum+unlabelNum)<0)==0)
    Y=Y(1:trial,:);
    S=S(1:trial,:);
    M=M(1:trial);
else
    [predictBest,tmp1,valuesBest,modelBest]=localDescent1(instance,ysvm,labelNum,unlabelNum,gamma,C,beta,alpha);
    Y(trial+1,:)=predictBest;
    S(trial+1,1)=sum(predictBest(labelNum+1:labelNum+unlabelNum)==trainLabel)/length(trainLabel);
    S(trial+1,2)=modelBest.obj;
    V(trial+1,:)=valuesBest;
    M{trial+1}=modelBest;
end

for i=1:trial
    
    disp(i);
    %对未标记样本生成随机标记,前80%次随机，后20%次在SVM的结果附近扰动
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
    [predictBest,tmp1,valuesBest,modelBest]=localDescent1(instance,labelNew,labelNum,unlabelNum,gamma,C,beta,alpha);
    
    %记录预测值
    Y(i,:)=predictBest;
    
    %记录精度和obj值
    S(i,1)=sum(predictBest(labelNum+1:labelNum+unlabelNum)==trainLabel)/length(trainLabel);
    S(i,2)=modelBest.obj;
    V(i,:)=valuesBest;
    M{i}=modelBest;
end

%用K-means对做出的101(也可能是100)个预测做聚类
[IDX,tmp1,tmp2,D]=kmeans(Y,clusterNum,'Distance','cityblock','EmptyAction','drop');

%排除空簇，算出最终的簇的个数
D=sum(D,1);
clusterIndex=find(isnan(D)==0);
clusterNum=size(find(isnan(D)==0),2);

%记录每个簇中最小的obj值
obj=zeros(clusterNum,1);

%记录所有的预测结果，每个簇对应一个，一列对应一个簇
prediction=zeros(labelNum+unlabelNum,clusterNum);
%记录所有的预测值，每个簇对应一个，一列对应一个簇
values=zeros(labelNum+unlabelNum,clusterNum);
%记录每个簇的预测的精度
accuracy=zeros(clusterNum,1);
model=cell(clusterNum,1);

for i=1:clusterNum
    %找到第i个簇对应的索引值
    index=find(IDX==clusterIndex(i));
    
    %将第i个簇对应的信息都提取出来
    tempS=S(index,:);
    tempY=Y(index,:);
    tempV=V(index,:);
    tempM=M(index);
    
    %obj2记录第i个簇所有预测的obj值
    obj2=tempS(:,2);
    
    %选出最好的，将信息保存下来
    [tmp,index2]=max(obj2);
    accuracy(i)=tempS(index2,1); 
    values(:,i)=tempV(index2,:); 
    obj(i)=tempS(index2,2);
    prediction(:,i)=tempY(index2,:)';
    model{i}=tempM(index2); 
end

train_time = cputime - start;

%s4vm的结果
start = cputime;

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

test_time = cputime - start;

% [s4vm(1),s4vm(2),s4vm(3)]=calculateAccAucF1(label,groundTruth);

% save(['result\temp\',datasetName,'_',kernel,'_',num2str(trialNum),'.mat'],'Y','ysvm','groundTruth','S','V','M');

end
