
load heart_scale.mat;

% the third term is the upper bound for each dual variable
model = fitcsvm(heart_scale_label, heart_scale_inst, rand(length(heart_scale_label),1), '-g 0.07');

[predict_label, accuracy, dec_values] = svmpredict(heart_scale_label, heart_scale_inst, model);