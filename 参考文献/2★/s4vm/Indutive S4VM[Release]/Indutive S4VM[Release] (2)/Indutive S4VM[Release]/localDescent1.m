function [predictLabel,acc,values,model]=localDescent1(instance,label,labelNum,unlabelNum,gamma,C,beta,alpha)
%LOCALDESCENT Summary of this function goes here
%   Detailed explanation goes here

predictLabelLastLast=label;
if(gamma==0)
    model=svmtrain(predictLabelLastLast,instance,C,'-t 0');
else
    model=svmtrain(predictLabelLastLast,instance,C,['-g ',num2str(gamma)]);
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
for i=(h1+1):(labelNum+unlabelNum-h2)
    if(valuesSort(i)>=0)
        predictLabel(index(i))=1;
    else
        predictLabel(index(i))=-1;
    end
end
predictLabelLast=predictLabel;
modelLast=model;

%generate a vector change of which 80% is 1 and rest is 0
count=1;
num=ceil(unlabelNum*0.2);
change=ones(unlabelNum,1);
v=zeros(num,1);
while(count<=unlabelNum*0.2)
    temp=ceil(rand*unlabelNum);
    flag=1;
    for j=1:count-1
        if(v(j)==temp)
            flag=0;
            break;
        end
    end
    if(flag==1)
        v(count)=temp;
        change(temp)=0;
        count=count+1;
    end
end
change=[ones(labelNum,1);change];
clear v;

%iterative
stop=0;
numIterative=0;
while(stop==0)
    labelNew=change.*predictLabelLast+(1-change).*predictLabelLastLast;
    if(gamma==0)
        model=svmtrain(labelNew,instance,C,'-t 0');
    else
        model=svmtrain(labelNew,instance,C,['-g ',num2str(gamma)]);
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
    for i=(h1+1):(labelNum+unlabelNum-h2)
        if(valuesSort(i)>=0)
            predictLabel(index(i))=1;
        else
            predictLabel(index(i))=-1;
        end
    end
    
    if((sum(predictLabel==predictLabelLast)==labelNum+unlabelNum&&model.obj==modelLast.obj)||numIterative>200)
        stop=1;
    else
        modelLast=model;
        predictLabelLastLast=predictLabelLast;
        predictLabelLast=predictLabel;
        %generate a vector change of which 80% is 1 and rest is 0
        count=1;
        num=ceil(unlabelNum*0.2);
        change=ones(unlabelNum,1);
        v=zeros(num,1);
        while(count<=unlabelNum*0.2)
            temp=ceil(rand*unlabelNum);
            flag=1;
            for j=1:count-1
                if(v(j)==temp)
                    flag=0;
                    break;
                end
            end
            if(flag==1)
                v(count)=temp;
                change(temp)=0;
                count=count+1;
            end
        end
        change=[ones(labelNum,1);change];
    end
end
end
