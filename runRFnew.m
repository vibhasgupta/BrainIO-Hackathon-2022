function [ I,MDL,Y,Score,Acc,I_train,Y_train,Score_train,Acc_train ] = runRFnew( trnLbl, trn, tstLbl, tst,numTrees, bag_ada  )
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here
% RandomForestTest
if strcmp(bag_ada,'ada')
        accuracy=0;
        for t=1:numTrees
            if length(unique(trnLbl))>2
                Mdl=fitensemble(trn,trnLbl,'adaboostm2',t,'tree','type','classification');%,'Crossval','on');
            else
                Mdl=fitensemble(trn,trnLbl,'adaboostm1',t,'tree','type','classification');%,'Crossval','on');
            end
            % Mdl = TreeBagger(i,trn,trnLbl,'method','classification','MinLeaf',5,'InBagFraction',0.2) ;
            [Yfit,score] = predict(Mdl,tst);
            Accuracy=sum(Yfit==tstLbl)/length(tstLbl);
            score=softmax(score')';
            if Accuracy>accuracy
                accuracy=Accuracy;
                I=t;
                MDL=Mdl;
                Y=Yfit;
                Score=score;
                Acc=Accuracy;
                Params={'adaboost Iterations:',t,'Accuracy:',Accuracy};

                disp(Params);
                
                [Yfit_train,score_train] = predict(Mdl,trn);
                Accuracy_train=sum(Yfit_train==trnLbl)/length(trnLbl);
                score_train=softmax(score_train')';
                I_train=t;
                Y_train=Yfit_train;
                Score_train=score_train;
                Acc_train=Accuracy_train;
            end


        end
else
        accuracy=0;
        for t=1:numTrees
            
            Mdl=fitensemble(trn,trnLbl,'bag',t,'tree','type','classification');%,'Crossval','on');
            
            % Mdl = TreeBagger(i,trn,trnLbl,'method','classification','MinLeaf',5,'InBagFraction',0.2) ;
            [Yfit,score] = predict(Mdl,tst);
            Accuracy=sum(Yfit==tstLbl)/length(tstLbl);
            score=softmax(score')';
            if Accuracy>accuracy
                accuracy=Accuracy;
                I=t;
                MDL=Mdl;
                Y=Yfit;
                Score=score;
                Acc=Accuracy;
                Params={'bag Iterations:',t,'Accuracy:',Accuracy};

                disp(Params);
                
                [Yfit_train,score_train] = predict(Mdl,trn);
                Accuracy_train=sum(Yfit_train==trnLbl)/length(trnLbl);
                score_train=softmax(score_train')';
                I_train=t;
                Y_train=Yfit_train;
                Score_train=score_train;
                Acc_train=Accuracy_train;
            end


        end
end
