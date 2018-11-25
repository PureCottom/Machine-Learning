function results = SVM(x,y,label)

tt = cputime;
addpath('liblinear-1.51-objective');
addpath('liblinear-1.51-objective\matlab');

labelindex = find(y ~= 0); 
x_train = x(labelindex,:);
y_train = y(labelindex);

opt = ['-s 3 -c 1'];
model = train(y_train,x_train, opt);

unlabelindex = find(y == 0); 
x_test = x(unlabelindex,:);
y_test = label(unlabelindex);
[predicted_label, accuracy, decision_values] = predict(y_test,x_test,model);
if y_train(1) == -1
    decision_values = -decision_values;
end

auc = evaluation_AUC(decision_values, y_test);

results.accuracy = accuracy(1)/100;
results.auc = auc;
results.cputime = cputime-tt;