function output=normalization(X)
%将X进行归一化到0到1，一行一个样本，一列对应一个维度
[instanceNum,dim]=size(X);
output=X;
for k=1:dim
    %找出每一维的最大值和最小值，通过一个线性映射将最大值映射到1，将最小值映射到0
    minNum=min(X(:,k));
    maxNum=max(X(:,k));
    if(minNum==maxNum)
        output(:,k)=zeros(instanceNum,1);
    else
        output(:,k)=(X(:,k)-minNum)/(maxNum-minNum);
    end
end

