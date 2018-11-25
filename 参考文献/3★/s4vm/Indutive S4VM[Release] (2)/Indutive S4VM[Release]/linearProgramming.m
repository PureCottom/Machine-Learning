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
