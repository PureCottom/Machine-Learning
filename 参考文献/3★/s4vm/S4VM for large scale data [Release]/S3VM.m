function results = S3VM(x,y,label)

tt = cputime;
addpath('liblinear-1.51-objective');
addpath('liblinear-1.51-objective\matlab');

%% obtain the multiple low-density seperators

% obtain SVM's decision values as the baseline
de_values = SVM_descision_values(x,y);

% rounding predicted labels by the balance constraint
predicted_label = rounding_labels(de_values,y);

unlabelindex = find(y == 0); 
y_tmp = y;
y_tmp(unlabelindex) = predicted_label;
[cand_y, decision_values] = local_search(x,y,y_tmp);

results.acc = sum(cand_y(unlabelindex) == label(unlabelindex))/length(unlabelindex);
results.auc = evaluation_AUC(decision_values, label(unlabelindex));
results.cputime = cputime-tt;
end


function [cand_y,dc] = local_search(x,y,y_tmp)

labelindex = find(y ~= 0);
unlabelindex = find(y == 0);

flag = 1;
iter = 1;

while flag && iter <= 5

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
    
    dc = decision_values; 
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