function result=myttest(a,b)
%返回1表示b显著好于a，返回-1表示a显著好于b，否则返回0

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

