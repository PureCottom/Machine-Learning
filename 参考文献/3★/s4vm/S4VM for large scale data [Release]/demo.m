function demo()

% clear; clc;

dataname = 'mnist7vs9';


str = [dataname '.mat'];
load(str);

str = [dataname '-split.mat'];
load(str);

results_SVM = cell(1,30);
results_S3VM = cell(1,30);
results_S4VM = cell(1,30);

for i = 1:30
    
    y = zeros(length(label),1);
    y(splitmatrix(i,:)) = label(splitmatrix(i,:));
    
    % SVM
    results_SVM{i} = SVM(data,y,label);

    % S3VM
    results_S3VM{i} = S3VM(data,y,label);
         
    % S4VM
    results_S4VM{i} = S4VM(data,y,label);

end

save([dataname '_results.mat'],'results_SVM','results_S3VM','results_S4VM');