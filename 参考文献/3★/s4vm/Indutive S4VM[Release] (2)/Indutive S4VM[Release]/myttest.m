function result=myttest(a,b)
%����1��ʾb��������a������-1��ʾa��������b�����򷵻�0

result=0;
h=ttest(a,b);
if(h==1)
    if(mean(a)<=mean(b))
        result=1;
    else
        result=-1;
    end
end

end

