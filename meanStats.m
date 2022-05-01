function ACC_test = meanStats(ACC_test,k)
    z=cell2mat(ACC_test(:,6:10));
     mn=mean(z);
     sd=std(z);

     for a=1:length(mn)
         ACC_test{k+1,a+5}=mn(a);
         ACC_test{k+2,a+5}=sd(a);
     end
      ACC_test(2:end+1,:)=ACC_test(1:end,:);
      ACC_test{1,1}='I';
      ACC_test{1,2}='Mdl';
      ACC_test{1,3}='GT';
      ACC_test{1,4}='Pred';
      ACC_test{1,5}='Score';
      ACC_test{1,6}='AvgAccuracy';
      ACC_test{1,7}='AvgPrecision';
      ACC_test{1,8}='AvgRecall';
      ACC_test{1,9}='AvgF1';
      ACC_test{1,10}='AvgWghtF1';
      ACC_test{k+2,1}='mean';
      ACC_test{k+3,1}='std';
end

