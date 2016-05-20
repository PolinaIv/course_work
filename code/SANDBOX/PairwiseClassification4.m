function  PairwiseClassification4(data_state,TrainingFraction,Dur,StepRange,Ncomp)
    Nc = length(data_state);
    ij = 1;
    RefTable = [];
     LTIndex = [];
     Nch = size(data_state{1},1);
     
    iijj = 0;
    for ii=1:Nch
        for jj=1:Nch
            iijj = iijj+1;
            if(ii>=jj)
                LTIndex = [LTIndex iijj];
            end;
        end;
    end;
    hy = figure;
    hz = figure;
    k = 0;
    for i = 1:Nc
       for j = 1:Nc
           k = k+1;
           if(i==j) 
               continue; 
           end;
           

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
            data_1_train =normcol(data_1_train);
            data_2_train =normcol(data_2_train);
           
            
            C10= data_1_train*data_1_train'/size(data_1_train,2);
            C20 = data_2_train*data_2_train'/size(data_2_train,2);
            nchan = size(C10,1);
            reg = 1;
            C1 = C10 + (0.01*reg+0.01) * trace(C10) * eye(nchan) / size(C10,1);
            C2 = C20 + (0.01*reg+0.01) * trace(C20) * eye(nchan) / size(C20,1);
            [V, d] = eig(C1,C2);
            M = [V(:,1)  V(:,end)]';
            
            data_1_test = normcol(data_1(:,ind_test1));
            data_2_test = normcol(data_2(:,ind_test2));
            data_1_test = (data_1(:,ind_test1));
            data_2_test = (data_2(:,ind_test2));
           
            y1 = M*data_1_train;
            y2 = M*data_2_train;
            z1 = M*data_1_test;
            z2 = M*data_2_test;
           
            
            figure(hy)
            subplot(6,6,k)
            plot(y1(1,:),y1(2,:),'.');hold on;
            plot(y2(1,:),y2(2,:),'r.');
            
            figure(hz)
            subplot(6,6,k)
            plot(z1(1,:),z1(2,:),'.');hold on;
            plot(z2(1,:),z2(2,:),'r.');
      
            
       end
    end;
  