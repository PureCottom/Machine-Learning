function results = S4VM(x,y,label)

tt = cputime;
addpath('liblinear-1.51-objective');
addpath('liblinear-1.51-objective\matlab');

%% obtain the multiple low-density seperators

% obtain SVM's decision values as the baseline
de_values = SVM_descision_values(x,y);

% rounding predicted labels by the balance constraint
predicted_label = rounding_labels(de_values,y);

unlabelindex = find(y == 0); 
N = 30; % number of candidate label assignments
for t = 1:N
    rp = randperm(length(predicted_label));
    N_flip = round(length(predicted_label)*0.2);
    y_tmp = y;
    y_tmp(unlabelindex) = predicted_label;
    ind = unlabelindex(rp(1:N_flip));
    y_tmp(ind) = -y_tmp(ind);
    [cand_y(:,t),obj(t)] = local_search(x,y,y_tmp);
end

% clustering candidate label assignments
cluster_num = 6;
[IDX] = kmeans(cand_y',cluster_num,'Distance','cityblock','EmptyAction','drop');

% get candidate low-density separators
for i = 1:cluster_num
    inx = find(IDX == i);
    if ~isempty(inx)
        [val,ix] = min(obj(inx));
        cand_y_final(:,i) = cand_y(:,inx(ix));
    end
end

%% construct S4VM
lambda = 3;
SVM_pre = sign(de_values);
L = length(SVM_pre);
T = size(cand_y_final,2);

f = zeros(1+L,1);
f(1+L) = 1;
A = zeros(T,L+1);
b = zeros(T,1);
for i = 1:T
    A(i,1:L) = -1/4*[(1+lambda)*cand_y_final(unlabelindex,i)'+(lambda-1)*SVM_pre'];
    A(i,L+1) = -1;
    b(i) = 1/4*(-(1+lambda)*SVM_pre'*cand_y_final(unlabelindex,i)+1-lambda);
end
lb = zeros(L+1,1);
ub = zeros(L+1,1);
lb(1:L) = -1; lb(L+1) = -inf;
ub(1:L) = 1; ub(L+1) = inf;
[s,fval] = linprog(f,A,b,[],[],lb,ub);
ss = sign(s(1:L));
if min(-A(:,1:L)*ss+b) > 0
    y = ss;
    dc = s(1:L);
else
    y = SVM_pre;
    dc = de_values;
end

results.acc = sum(y == label(unlabelindex))/L;
results.auc = evaluation_AUC(dc, label(unlabelindex));

% get S4VM_best
best_acc = 0;
for i = 1:N
    y = cand_y(:,i);
    cur_acc = sum(y(unlabelindex) == label(unlabelindex))/L;
    if cur_acc > best_acc 
        best_acc = cur_acc;
        best_y = y;
    end
end
results.acc_best = sum(best_y(unlabelindex) == label(unlabelindex))/L;
results.auc_best = evaluation_AUC(best_y(unlabelindex), label(unlabelindex));

% get S4VM_min
[val, ix] = min(obj);
y = cand_y(:,ix);
results.acc_min = sum(y(unlabelindex) == label(unlabelindex))/L;
results.auc_min = evaluation_AUC(y(unlabelindex), label(unlabelindex));

% get S4VM_com
y = sum(cand_y(:,ix),2);
results.acc_com = sum(sign(y(unlabelindex)) == label(unlabelindex))/L;
results.auc_com = evaluation_AUC(y(unlabelindex), label(unlabelindex));

results.cputime = cputime-tt;

end

function [cand_y,obj] = local_search(x,y,y_tmp)

labelindex = find(y ~= 0);
unlabelindex = find(y == 0);

flag = 1;
iter = 1;

while flag && iter <= 5

%     xx = x;    
%     xx(labelindex,:) = diag(y(labelindex))*x(labelindex,:);
%     batchsize = 500;
%     fold = ceil(length(unlabelindex)/batchsize);
%     for t = 1:fold
%         if t < fold
%             ind = unlabelindex(batchsize*(t-1)+1:batchsize*t);        
%         else
%             ind = unlabelindex(batchsize*(t-1)+1:end);
%         end
%         xx(ind,:) = -diag(y_tmp(ind))*x(ind,:);
%     end
%     yy = zeros(length(y),1);
%     yy(labelindex) = 1;
%     yy(unlabelindex) = -1;
% 
%     if yy(1) == 1
%         opt = ['-c 1 -w2 0.1'];
%     else
%         opt = ['-c 1 -w1 0.1'];
%     end
    opt = ['-s 3 -c 1'];
    model = train(y_tmp,x,opt);
    [predicted_label, accuracy, decision_values] = predict(y_tmp(unlabelindex),x(unlabelindex,:),model);
    if y_tmp(1) == -1
        decision_values = - decision_values;
    end
    predicted_label = rounding_labels(decision_values,y);
    
    if norm(predicted_label' - y_tmp(unlabelindex),2) == 0
        flag = 0;
        
    else
        tmp_y = y_tmp;
        tmp_y(labelindex) = y(labelindex);
        tmp_y(unlabelindex) = predicted_label;
        y_tmp = tmp_y;
    end
    
    obj = model.obj; 
    iter = iter + 1;

end

cand_y = y_tmp;
end


function predicted_label = rounding_labels(de_values,y)

pos = sum(y == 1);
neg = sum(y == -1);
alpha = pos/(pos+neg);

[value, inx] = sort(de_values,'descend');
lpos = round(alpha*length(de_values));
lneg = length(de_values) - lpos;
predicted_label(inx(1:lpos)) = 1;
predicted_label(inx(lpos+1:lpos+lneg)) = -1;

end

function de_values = SVM_descision_values(x,y)

labelindex = find(y ~= 0); 
x_train = x(labelindex,:);
y_train = y(labelindex);

opt = ['-s 3 -c 1'];
model = train(y_train,x_train,opt);

unlabelindex = find(y == 0); 
x_test = x(unlabelindex,:);
y_test = y(unlabelindex);
[predicted_label, acc, de_values] = predict(y_test,x_test,model);
if y_train(1) == -1
    de_values = - de_values;
end

end