function collect_results_cputime

clear; clc;
labelN = 10;

%%%%%%%%%%  UCI  %%%%%%%%%%%%%%%%%
datanames = [{'austra'},{'australian'}, {'breastw'},{'clean1'},{'diabetes'},...
    {'haberman'},{'heart'},{'house-votes'},{'house'},{'ionosphere'},...
    {'isolet'},{'liverDisorders'},{'optdigits'},{'vehicle'},{'wdbc'}];

for i = 1:length(datanames)
    dataset = datanames{i};
    str = ['cpu_time\UCI_b_' dataset '_S3VM_' num2str(labelN) '.mat'];
    load(str);
    
    str = ['cpu_time\UCI_b_' dataset '_S4VM_' num2str(labelN) '.mat'];
    load(str);
    
    LinearS3VM_train(i) = mean(LinearS3VM_train_time);
    LinearS3VM_test(i) = mean(LinearS3VM_test_time);
    
    LinearS4VM_train(i) = mean(LinearS4VM_train_time);
    LinearS4VM_test(i) = mean(LinearS4VM_test_time);
    
    
   
end

train_time = [LinearS3VM_train',LinearS4VM_train'];
test_time = [LinearS3VM_test',LinearS4VM_test'];
str = ['cpu_time\total_result.mat'];
save(str,'train_time','test_time');
%%%%%%%%%%%  Benchmark  %%%%%%%%%%%%%%%%%

% datanames = [{'1'},{'2'},{'3'},{'4'},{'5'}];
% realnames = [{'digit1'},{'usps'},{'coil'},{'bci'},{'g241c'}];
% 
% for i = 1:length(datanames)
%     dataset = datanames{i};
%     str = ['result\SSL_b_SSL,set=' dataset '_S3VM_' num2str(labelN) '.mat'];
%     load(str);
%     
%      SVM = resultLinearsvm(1,:)';
%     TSVM = resultLinears3vm(1,:)';
%     
%     SVM_rbf = resultRBFsvm(1,:)';
%     TSVM_rbf = resultRBFs3vm(1,:)';
%     
%     str = [realnames{i} ' & '];
%     str = [str my_num2str(mean(SVM)) ' $\pm$ ' my_num2str(std(SVM)) ' / '];
%     str = [str my_num2str(mean(SVM_rbf)) ' $\pm$ ' my_num2str(std(SVM_rbf)) ' & '];    
%     str = [str result2str(SVM, TSVM) ' / '];
%     str = [str result2str(SVM_rbf, TSVM_rbf) ' \\ '];
%     
%     disp(str)
%     
% end
% 






end

function str = my_num2str(v)

if v < 1
    v = v*100;
end

u = v*10;
u = round(u);
str = num2str(u/10);

end

function str = result2str(A, B)

[H,p] = ttest(A,B);
if H == 0
    str = [my_num2str(mean(B)) ' $\pm$ ' my_num2str(std(B))];
else
    if mean(A) > mean(B)
        str = ['\underline{' my_num2str(mean(B)) ' $\pm$ ' my_num2str(std(B)) '}'];
    else
        str = ['\textbf{' my_num2str(mean(B)) ' $\pm$ ' my_num2str(std(B)) '}'];
    end
end

end
