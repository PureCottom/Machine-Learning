% addpath('libsvm-mat-2.89-3-box constraint');
% datasetType='UCI';
% labelNum=20;
% kernel='RBF';
%
% if(strcmp(datasetType,'UCI'))
%     datasetName={'austra','australian','breastw','clean1','diabetes','german','haberman','heart','house','house-votes','ionosphere','isolet','liverDisorders','optdigits','vehicle','wdbc'};
%     datasetNum=length(datasetName);
%     trialNum=30;
% else
%     datasetName={'SSL,set=1','SSL,set=2','SSL,set=3','SSL,set=4','SSL,set=5','SSL,set=7','SSL,set=9'};
%     datasetNum=length(datasetName);
%     trialNum=12;
% end
%
% C1=100;
% result=zeros(datasetNum,trialNum);
% ySVM=cell(datasetNum,1);
% for k=20:20:80
%     for i=1:datasetNum
%         i
%         load(['dataset/',datasetType,'/',datasetName{i},'.mat']);
%         load(['dataset/',datasetType,'/',datasetName{i},',splits,labeled=',num2str(labelNum),'.mat']);
%         [instanceNum,dim]=size(X);
%         X=normalization(X);
%
%         Y=[];
%         for j=1:trialNum
%             index=idxLabs(j,:);
%             labelInstance=X(index,:);
%             label=y(index,:);
%             if(k==20)
%                 index=idxUnls20(j,:);
%             end
%             if(k==40)
%                 index=idxUnls40(j,:);
%             end
%             if(k==60)
%                 index=idxUnls60(j,:);
%             end
%             if(k==80)
%                 index=idxUnls80(j,:);
%             end
%             unlabelInstance=X(index,:);
%             groundTruth=y(index,:);
%
%             labelNum=length(label);
%             C=ones(labelNum,1)*C1;
%
%             if(strcmp(kernel,'RBF'))
%                 gammaSVM=length(pdist(X))/sum(pdist(X));
%                 model=svmtrain(label,labelInstance,C,['-g ',num2str(gammaSVM)]);
%             else
%                 model=svmtrain(label,labelInstance,C,'-t 0');
%             end
%
%             [predictLabel,acc,values]=svmpredict(groundTruth,unlabelInstance,model);
%             result(i,j)=acc(1,1);
%             Y=[Y predictLabel];
%         end
%         ySVM{i}=Y;
%     end
%     save(['result\',datasetType,'_SVM_',num2str(labelNum),'_',kernel,'_',num2str(k),'.mat'],'result','ySVM');
% end
%


addpath('libsvm-mat-2.89-3-box constraint');
datasetType='UCI';
labelNum=20;
kernel='RBF';

if(strcmp(datasetType,'UCI'))
    datasetName={'austra','australian','breastw','clean1','diabetes','german','haberman','heart','house','house-votes','ionosphere','isolet','liverDisorders','optdigits','vehicle','wdbc'};
    datasetNum=length(datasetName);
    trialNum=30;
else
    datasetName={'SSL,set=1','SSL,set=2','SSL,set=3','SSL,set=4','SSL,set=5','SSL,set=7','SSL,set=9'};
    datasetNum=length(datasetName);
    trialNum=12;
end

C1=100;
result=zeros(datasetNum,trialNum);
ySVM=cell(datasetNum,1);

for i=1:datasetNum
    i
    load(['dataset/',datasetType,'/',datasetName{i},'.mat']);
    load(['dataset/',datasetType,'/',datasetName{i},',splits,labeled=',num2str(labelNum),'.mat']);
    [instanceNum,dim]=size(X);
    X=normalization(X);
    
    Y=[];
    for j=1:trialNum
        index=idxLabs(j,:);
        labelInstance=X(index,:);
        label=y(index,:);
        index=idxUnls(j,:);
        unlabelInstance=X(index,:);
        groundTruth=y(index,:);
        
        labelNum=length(label);
        C=ones(labelNum,1)*C1;
        
        if(strcmp(kernel,'RBF'))
            gammaSVM=length(pdist(X))/sum(pdist(X));
            model=svmtrain(label,labelInstance,C,['-g ',num2str(gammaSVM)]);
        else
            model=svmtrain(label,labelInstance,C,'-t 0');
        end
        
        [predictLabel,acc,values]=svmpredict(groundTruth,unlabelInstance,model);
        result(i,j)=acc(1,1);
        Y=[Y predictLabel];
    end
    ySVM{i}=Y;
end
save(['result\',datasetType,'_SVM_',num2str(labelNum),'_',kernel,'.mat'],'result','ySVM');



