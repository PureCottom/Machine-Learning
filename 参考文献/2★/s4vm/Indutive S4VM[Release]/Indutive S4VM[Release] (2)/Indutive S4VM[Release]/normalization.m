function output=normalization(X)
%��X���й�һ����0��1��һ��һ��������һ�ж�Ӧһ��ά��
[instanceNum,dim]=size(X);
output=X;
for k=1:dim
    %�ҳ�ÿһά�����ֵ����Сֵ��ͨ��һ������ӳ�佫���ֵӳ�䵽1������Сֵӳ�䵽0
    minNum=min(X(:,k));
    maxNum=max(X(:,k));
    if(minNum==maxNum)
        output(:,k)=zeros(instanceNum,1);
    else
        output(:,k)=(X(:,k)-minNum)/(maxNum-minNum);
    end
end

