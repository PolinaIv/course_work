function [ Res A] = PairwiseClassification(data_state,TrainingFraction,Dur,StepRange,Ncomp)
    Nc = length(data_state);
    ij = 1;
    RefTable = [];
    for i = 1:Nc
       for j = 1:Nc
           if(i==j) continue; end;

            RefTable(ij,:) = [ij i j];

            disp(i);
            disp(j);  

            data_1 = data_state{i};
            data_2 = data_state{j};

            ind_train1 = 1:fix(size(data_1,2)*TrainingFraction);
            ind_train2 = 1:fix(size(data_2,2)*TrainingFraction);

            ind_test1 = fix(size(data_1,2)*TrainingFraction)+1:size(data_1,2);
            ind_test2 = fix(size(data_2,2)*TrainingFraction)+1:size(data_2,2);

            data_1_train = data_1(:,ind_train1);
            data_2_train = data_2(:,ind_train2);

            data_1_test = data_1(:,ind_test1);
            data_2_test = data_2(:,ind_test2);

            C10 = data_1_train * data_1_train' / size(data_1_train,2);
            C20 = data_2_train * data_2_train' / size(data_2_train,2);

            nchan = size(C10,1);
            A(ij,:) = zeros(1,20);
            clear QQQ1trn QQQ1tst QQQ2trn QQQ2tst
            for reg = 1:10
                C1 = C10 + (0.01*reg+0.01) * trace(C10) * eye(nchan) / size(C10,1);
                C2 = C20 + (0.01*reg+0.01) * trace(C20) * eye(nchan) / size(C20,1);

                [V d] = eig(C1,C2);

                M = V(:,[1:Ncomp, end-Ncomp+1:end])';

                MM{reg} = M;

                Y1 = M * data_1_train;
                Y2 = M * data_2_train;

                LTIndex = [];
                iijj = 0;
                for ii=1:size(Y1,1)
                    for jj=1:size(Y1,1)
                        iijj = iijj+1;
                        if(ii>=jj)
                            LTIndex = [LTIndex iijj];
                        end;
                    end;
                end;
                clear R1 R2
                range = 1:Dur;
                t = 1;
                while(range(end)<size(Y1,2))
                     R = Y1(:,range)*Y1(:,range)'/length(range);
                     R1(:,t) = R(LTIndex);
                     range = range+StepRange;
                     t = t+1;
                end;

                range = 1:Dur;
                t = 1;
                while(range(end)<size(Y2,2))
                     R = Y2(:,range)*Y2(:,range)'/length(range);
                     R2(:,t) = R(LTIndex);
                     range = range+StepRange;
                     t = t+1;
                end;

                y_data_train = [R1 R2];
                y_states_train = [ones(1,size(R1,2)),2*ones(1,size(R2,2))];

                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                [C,err,P,logp,coeff] = classify(y_data_train', y_data_train', y_states_train,'mahalanobis');
   
                Y1 = M * data_1_test;
                Y2 = M * data_2_test;
                
                clear R1 R2
                range = 1:Dur;
                t = 1;
                while(range(end)<size(Y1,2))
                     R = Y1(:,range)*Y1(:,range)'/length(range);
                     R1(:,t) = R(LTIndex);
                     range = range+StepRange;
                     t = t+1;
                end;

                range = 1:Dur;
                t = 1;
                while(range(end)<size(Y2,2))
                     R = Y2(:,range)*Y2(:,range)'/length(range);
                     R2(:,t) = R(LTIndex);
                     range = range+StepRange;
                     t = t+1;
                end;

                y_data_test = [R1 R2];
                y_states_test = [ones(1,size(R1,2)),2*ones(1,size(R2,2))];
                
                [C,err,P,logp,coeff] = classify(y_data_test', y_data_train', y_states_train,'mahalanobis');
                   
                           
                th = 0.5*(mean(Q0trn)+mean(Q1trn));
                P = (sum(Q0tst>th)+sum(Q1tst<th)) /(length(Q1tst) + length(Q0tst));
                
                A(ij,reg) = P;
                QQQ1trn(reg,:) = Q0trn;
                QQQ2trn(reg,:) = Q1trn;
                QQQ1tst(reg,:) = Q0tst;
                QQQ2tst(reg,:) = Q1tst;
             end %reg\
             
             if(i==3 & j==4)
                 op = 1;
             end;

             [maxv,maxind] = max(A(ij,:));
             Res{i,j}.ROC = ResultsReg(maxind);
             Res{i,j}.M = MM{maxind};
             Res{i,j}.W = WW{maxind};
             Res{i,j}.AUC = maxv;
             Res{i,j}.AUCvsReg = A;
             Res{i,j}.Qtrn1 = QQQ1trn(maxind,:);
             Res{i,j}.Qtrn2 = QQQ2trn(maxind,:);
             Res{i,j}.Qtst1 = QQQ1tst(maxind,:);
             Res{i,j}.Qtst2 = QQQ2tst(maxind,:);
             
             
             ij = ij+1;
        end %j
    end %i
end

