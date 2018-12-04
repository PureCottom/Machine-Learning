function [train_time, test_time] = segments_cputime(X,Y,X_test,opt)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% X : dim * N
% Y : N * 1
% X_test: dim * M
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

start = cputime;
posy =(Y == 1);
negy =(Y == -1);
uny = (Y == 0);
dim = size(X,1);
positive = X(1:end,posy);
negative = X(1:end,negy);
zerotive = X(1:end,uny);

% д��ʱ�ļ� ��дpositive, һ��һ�е�д
fid = fopen('traintmp', 'w');
for ll = 1:size(positive,2);
    textl = [''];
    for mm = 1:dim
        textl = [textl ' ' num2str(mm) ':' num2str(positive(mm,ll))];
    end
    textl = ['+1' textl];
    fprintf(fid,'%s\n',textl);
end
for ll = 1:size(negative,2);
    textl = [''];
    for mm = 1:dim
        textl = [textl ' ' num2str(mm) ':' num2str(negative(mm,ll))];
    end
    textl = ['-1' textl];
    fprintf(fid,'%s\n',textl);
end
for ll = 1:size(zerotive,2);
    textl = [''];
    for mm = 1:dim
        textl = [textl ' ' num2str(mm) ':' num2str(zerotive(mm,ll))];
    end
    textl = ['0' textl];
    fprintf(fid,'%s\n',textl);
end
fclose(fid);

% ѵ��svm
if ~isfield(opt,'g')
    str=['svm_learn -t ',num2str(opt.t),' -c ',num2str(opt.c),' traintmp modeltmp > trainoutput.txt'];
else
    str=['svm_learn -t ',num2str(opt.t),' -c ',num2str(opt.c),' -g ' num2str(opt.g),' traintmp modeltmp > trainoutput.txt'];
end
system(str);

train_time = cputime - start;

start = cputime;
fid = fopen('testtmp', 'w');
for ll = 1:size(positive,2);
    textl = [''];
    for mm = 1:dim
        textl = [textl ' ' num2str(mm) ':' num2str(positive(mm,ll))];
    end
    textl = ['0' textl];
    fprintf(fid,'%s\n',textl);
end
for ll = 1:size(negative,2);
    textl = [''];
    for mm = 1:dim
        textl = [textl ' ' num2str(mm) ':' num2str(negative(mm,ll))];
    end
    textl = ['0' textl];
    fprintf(fid,'%s\n',textl);
end
for ll = 1:size(X_test,2);
    textl = [''];
    for mm = 1:dim
        textl = [textl ' ' num2str(mm) ':' num2str(X_test(mm,ll))];
    end
    textl = ['0' textl];
    fprintf(fid,'%s\n',textl);
end
fclose(fid);
% ����svm ���ﻹ��һЩ����
% Ӧ�ò������
!svm_classify testtmp modeltmp precision > testoutput.txt
% read precision
dec_values_tsvm = csvread('precision');
label = sign(dec_values_tsvm);
label(1:10) = [];
!del testtmp traintmp modeltmp precision 
test_time = cputime - start;
