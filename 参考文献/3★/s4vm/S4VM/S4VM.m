function prediction=S4VM(labelInstance,label,unlabelInstance,kernel,C1,C2,sampleTime,gamma)

% S4VM implements the S4VM algorithm in [1].
%  ========================================================================
%
%  Input:
%  S4VM takes 8 input parameters(the first three parameters are necessary,
%  the rest are optional), in this order:
%
%  labelInstance: a matrix with size labelInstanceNum * dimension. Each row
%                 vector of labelInstance is a instance vector, of which
%                 the label is already known.
%
%  label: a column binary vector with length labelInstanceNum. Each element
%         is +1 or -1 and the jth element is the label of the jth row
%         vector of labelInstance.
%
%  unlabelInstance: a matrix with size unlabelInstanceNum * dimension. Each
%                   row vector of unlabeledInstance is a instance vector,
%                   of which the label is still unknown.
%
%  kernel: the kernel used for the S4VM with two options: 'RBF' or 'Linear'.
%          Default value is 'RBF'.
%
%  C1: weight for the hinge loss of labeled instance. Default value is 100.
%
%  C2: weight for the hinge loss of unlabeled instance. If C2 is set as 0,
%      our S4VM will degenerate to standard SVM. Default value is 0.1
%
%  sampleTime: the sampling times for each sampleTime. Default value is 100.
%
%  gamma: parameter gamma is the width of RBF kernel. Default value is
%         average distance between instances.
%
%  In our paper, all the features of the instances are normalized to [0,1]
%
%  ========================================================================
%
%  Output:
%  prediction: Since our S4VM method is transductive, the output of S4VM is the
%              predicated labels of unlabeled instances.
%
%  ========================================================================
%
%  Example:
%    prediction=S4VM(l,y,u,'RBF');
%    prediction=S4VM(l,y,u,'RBF',100,1);
%    prediction=S4VM(l,y,u,'Linear',100,0.1,100,0.25);
%
%  ========================================================================
%
%  Reference:
%  [1]  Yu-Feng Li and Zhi-Hua Zhou. Towards Making Unlabeled Data Never Hurt. In: Proceedings of the 28th International Conference on Machine Learning (ICML'11), Bellevue, Washington, 2011.
%

labelNum=length(label);
unlabelNum=size(unlabelInstance,1);
instance=[labelInstance;unlabelInstance];

if(nargin<4)
    kernel='RBF';
end
if(nargin<5)
    C1=100;
end
if(nargin<6)
    C2=0.1;
end
if(nargin<7)
    sampleTime=100;
end
if(nargin<8)
    if(strcmp(kernel,'RBF'))
        gamma=length(pdist(instance))/sum(pdist(instance));
    end
end

C=[ones(labelNum,1)*C1;ones(unlabelNum,1)*C2];

beta=sum(label)/length(label);
alpha=0.1;
clusterNum=floor(sampleTime/10);
Y=zeros(sampleTime+1,labelNum+unlabelNum);
S=zeros(sampleTime+1,1);

if(strcmp(kernel,'Linear'))
    model=fitcsvm(label,labelInstance,ones(labelNum,1)*C1,'-t 0');
else
    model=fitcsvm(label,labelInstance,ones(labelNum,1)*C1,['-g ',num2str(gamma)]);
end
[ysvm,~,~]=svmpredict([label;ones(unlabelNum,1)],instance,model);

if(sum(ysvm(labelNum+1:labelNum+unlabelNum)>0)==0||sum(ysvm(labelNum+1:labelNum+unlabelNum)<0)==0)
    Y=Y(1:sampleTime,:);
    S=S(1:sampleTime);
else
    [predictBest,~,~,modelBest]=localDescent(instance,ysvm,labelNum,unlabelNum,gamma,C,beta,alpha);
    Y(sampleTime+1,:)=predictBest;
    S(sampleTime+1)=modelBest.obj;
end

for i=1:sampleTime
    if(i<=sampleTime*0.8)
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
    [predictBest,~,~,modelBest]=localDescent(instance,labelNew,labelNum,unlabelNum,gamma,C,beta,alpha);
    
    Y(i,:)=predictBest;
    S(i)=modelBest.obj;
end

[IDX,~,~,D]=kmeans(Y,clusterNum,'Distance','cityblock','EmptyAction','drop');
D=sum(D,1);
clusterIndex=find(isnan(D)==0);
clusterNum=size(find(isnan(D)==0),2);

prediction=zeros(labelNum+unlabelNum,clusterNum);

for i=1:clusterNum
    index=find(IDX==clusterIndex(i));
    tempS=S(index);
    tempY=Y(index,:);
    [~,index2]=max(tempS);
    prediction(:,i)=tempY(index2,:)';
end

% use linear programming to get the final prediction
prediction=linearProgramming(prediction,ysvm,labelNum,3);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    function label=linearProgramming(yp,ysvm,labelNum,lambda)
        yp(1:labelNum,:)=[];
        ysvm(1:labelNum,:)=[];
        [u,yNum]=size(yp);
        A=[ones(yNum,1) ((1-lambda)*repmat(ysvm,1,yNum)/4-(1+lambda)*yp/4)'];
        C=ones(yNum,1)*(1-lambda)*u/4-(1+lambda)*yp'*ysvm/4;
        g=[-1;zeros(u,1)];
        lb=[-inf;-ones(u,1)];
        ub=[inf;ones(u,1)];
        prediction=linprog(g,A,C,[],[],lb,ub);
        if(prediction(1)<0)
            label=ysvm;
        else
            prediction(1)=[];
            label=sign(prediction);
        end
    end

    function [predictLabel,acc,values,model]=localDescent(instance,label,labelNum,unlabelNum,gamma,C,beta,alpha)
        
        predictLabelLastLast=label;
        if(gamma==0)
            model=fitcsvm(predictLabelLastLast,instance,C,'-t 0');
        else
            model=fitcsvm(predictLabelLastLast,instance,C,['-g ',num2str(gamma)]);
        end
        [predictLabel,acc,values]=svmpredict(predictLabelLastLast,instance,model);%对未标记样本进行预测
        if(values(1)*predictLabel(1)<0)
            values=-values;
        end
        
        %update predictLabel
        [valuesSort,index]=sort(values,1,'descend');
        h1=ceil((labelNum+unlabelNum)*(1+beta-alpha)/2);
        h2=ceil((labelNum+unlabelNum)*(1-beta-alpha)/2);
        predictLabel(index(1:h1))=1;
        predictLabel(index(labelNum+unlabelNum-h2+1:labelNum+unlabelNum))=-1;
        valuesSort=valuesSort((h1+1):(labelNum+unlabelNum-h2));
        predictLabel(index(find(valuesSort>=0)+h1))=1;
        predictLabel(index(find(valuesSort<0)+h1))=-1;
        predictLabelLast=predictLabel;
        modelLast=model;
        
        %generate a vector change of which 80% is 1 and rest is 0
        num=ceil(unlabelNum*0.2);
        index=randperm(unlabelNum);
        index=index(1:num);
        change=ones(unlabelNum,1);
        change(index)=0;
        change=[ones(labelNum,1);change];
        
        %iterative
        stop=0;
        numIterative=0;
        while(stop==0)
            labelNew=change.*predictLabelLast+(1-change).*predictLabelLastLast;
            if(gamma==0)
                model=fitcsvm(labelNew,instance,C,'-t 0');
            else
                model=fitcsvm(labelNew,instance,C,['-g ',num2str(gamma)]);
            end
            [predictLabel,acc,values]=svmpredict(labelNew,instance,model);
            numIterative=numIterative+1;
            if(values(1)*predictLabel(1)<0)
                values=-values;
            end
            %update predictLabel
            [valuesSort,index]=sort(values,1,'descend');
            predictLabel(index(1:h1))=1;
            predictLabel(index(labelNum+unlabelNum-h2+1:labelNum+unlabelNum))=-1;
            valuesSort=valuesSort((h1+1):(labelNum+unlabelNum-h2));
            predictLabel(index(find(valuesSort>=0)+h1))=1;
            predictLabel(index(find(valuesSort<0)+h1))=-1;
            
            if((sum(predictLabel==predictLabelLast)==labelNum+unlabelNum&&model.obj==modelLast.obj)||numIterative>200)
                stop=1;
            else
                modelLast=model;
                predictLabelLastLast=predictLabelLast;
                predictLabelLast=predictLabel;
                %generate a vector change of which 80% is 1 and rest is 0
                num=ceil(unlabelNum*0.2);
                index=randperm(unlabelNum);
                index=index(1:num);
                change=ones(unlabelNum,1);
                change(index)=0;
                change=[ones(labelNum,1);change];
            end
        end
    end
end
