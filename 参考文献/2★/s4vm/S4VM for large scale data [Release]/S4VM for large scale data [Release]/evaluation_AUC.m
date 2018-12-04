function auc = evaluation_AUC(decision_values, label)

L = length(label);
pos = sum(label == 1);
neg = sum(label == -1);
error = 0;

for t = 1:10
    rp = randperm(L);
    dv = decision_values(rp);
    lb = label(rp);
    [value,inx] = sort(dv,'descend');
    
    flag = 0;

    for i = 1:L
        if lb(inx(i)) == -1
            flag = flag +1;
            error = error + (pos-i+flag);
        end
    end
end
error = error/10;
auc = 1-error/(pos*neg);

% another version
% error_new = 0;
% for i = 1:L
%     for j = i+1:L
%         if label(i) ~= label(j)
%             if (decision_values(i)-decision_values(j))*(label(i)-label(j)) < 0
%                 error_new = error_new +1;
%             end
%             if (decision_values(i)-decision_values(j)) == 0
%                 error_new = error_new + 1/2;
%             end
%         end
%     end
% end
% 
% auc_new = 1-error_new/(pos*neg);