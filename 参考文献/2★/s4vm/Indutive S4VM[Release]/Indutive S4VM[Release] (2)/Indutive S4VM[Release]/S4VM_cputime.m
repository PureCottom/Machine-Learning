function [train_time,test_time] = S4VM_cputime(instance,label,trainLabel,testInstance,groundTruth,labelNum,unlabelNum,gamma,C,trial,datasetName,trialNum,kernel)


start = cputime;

% best=zeros(1,3);
% minimum=zeros(1,3);
% com=zeros(1,3);
% s4vm=zeros(1,3);

beta=sum(label)/length(label);
alpha=0.1;
clusterNum=floor(trial/10);%����ĸ���
Y=zeros(trial+1,labelNum+unlabelNum);
S=zeros(trial+1,2);
V=zeros(trial+1,labelNum+unlabelNum);
M=cell(trial+1,1);

%ʹ��SVM��δ���������Ԥ����Ϊ��ʼ�½�����һ��localDescent
if(gamma==0)
    model=svmtrain(label,instance(1:labelNum,:),C(1:labelNum),'-t 0');
else
    model=svmtrain(label,instance(1:labelNum,:),C(1:labelNum),['-g ',num2str(gamma)]);
end
[ysvm,tmp1,tmp2]=svmpredict([label;trainLabel],instance,model);
[ysvmpre,tmp1,tmp2]=svmpredict(groundTruth,testInstance,model);

%���SVM������δ�����������Ϊһ�࣬�ͷ��������������
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
    %��δ�����������������,ǰ80%���������20%����SVM�Ľ�������Ŷ�
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
    
    %��¼Ԥ��ֵ
    Y(i,:)=predictBest;
    
    %��¼���Ⱥ�objֵ
    S(i,1)=sum(predictBest(labelNum+1:labelNum+unlabelNum)==trainLabel)/length(trainLabel);
    S(i,2)=modelBest.obj;
    V(i,:)=valuesBest;
    M{i}=modelBest;
end

%��K-means��������101(Ҳ������100)��Ԥ��������
[IDX,tmp1,tmp2,D]=kmeans(Y,clusterNum,'Distance','cityblock','EmptyAction','drop');

%�ų��մأ�������յĴصĸ���
D=sum(D,1);
clusterIndex=find(isnan(D)==0);
clusterNum=size(find(isnan(D)==0),2);

%��¼ÿ��������С��objֵ
obj=zeros(clusterNum,1);

%��¼���е�Ԥ������ÿ���ض�Ӧһ����һ�ж�Ӧһ����
prediction=zeros(labelNum+unlabelNum,clusterNum);
%��¼���е�Ԥ��ֵ��ÿ���ض�Ӧһ����һ�ж�Ӧһ����
values=zeros(labelNum+unlabelNum,clusterNum);
%��¼ÿ���ص�Ԥ��ľ���
accuracy=zeros(clusterNum,1);
model=cell(clusterNum,1);

for i=1:clusterNum
    %�ҵ���i���ض�Ӧ������ֵ
    index=find(IDX==clusterIndex(i));
    
    %����i���ض�Ӧ����Ϣ����ȡ����
    tempS=S(index,:);
    tempY=Y(index,:);
    tempV=V(index,:);
    tempM=M(index);
    
    %obj2��¼��i��������Ԥ���objֵ
    obj2=tempS(:,2);
    
    %ѡ����õģ�����Ϣ��������
    [tmp,index2]=max(obj2);
    accuracy(i)=tempS(index2,1); 
    values(:,i)=tempV(index2,:); 
    obj(i)=tempS(index2,2);
    prediction(:,i)=tempY(index2,:)';
    model{i}=tempM(index2); 
end

train_time = cputime - start;

%s4vm�Ľ��
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
