function [acc,auc,f1]=calculateAccAucF1(prediction,groundTruth)

%prediction=[-1 -1 -1 -1 -1 -1 1];
%groundTruth=[-1 -1 -1 -1 -1 1 1];

tp=length(find(prediction+groundTruth==2));
tn=length(find(prediction+groundTruth==-2));
fn=length(find(groundTruth==1))-tp;
fp=length(find(groundTruth==-1))-tn;

acc=(tp+tn)/(tp+tn+fp+fn);

num=0;
indexP=find(groundTruth==1);
indexN=find(groundTruth==-1);
for i=1:length(indexP)
    for j=1:length(indexN)
        if(prediction(indexP(i))>prediction(indexN(j)))
            num=num+1;
        end
        if(prediction(indexP(i))==prediction(indexN(j)))
            num=num+0.5;
        end
    end
end
auc=num/(length(indexP)*length(indexN));

if(tp+fp>0)
    precision=tp/(tp+fp);
else
    precision=0;
end
if(tp+fn>0)
    recall=tp/(tp+fn);
else
    recall=0;
end
if(precision+recall>0)
    f1=2*precision*recall/(precision+recall);
else
    f1=0;
end

end

