function [ Result] = ClassifyPairCov(data, indTr1,indTr2,Lambda,Dur,StepRange,Ncomp)
      
    % calculate class covariances\
    data_1_train = data(:,indTr1);
    data_2_train = data(:,indTr2);
    
    C10 = data_1_train * data_1_train' / size(data_1_train,2);
    C20 = data_2_train * data_2_train' / size(data_2_train,2);

    nchan = size(C10,1);
    
    %regularize covariances
    C1 = C10 + Lambda * trace(C10) * eye(nchan) / nchan;
    C2 = C20 + Lambda * trace(C20) * eye(nchan) / nchan;
    
    % do generalized eigenvalue decomp
    [V d] = eig(C1,C2);
    iV = inv(V);
    
    M12 = V(:,[1:Ncomp, end-Ncomp+1:end])';
  
    G12 = iV([1:Ncomp, end-Ncomp+1:end],:);
  
    Y1 = M12 * data_1_train;
    Y2 = M12 * data_2_train;
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
            
    %%%%%%%%%%%%
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

      % build shrinkage (linear) classifier
    obj = train_shrinkage(y_data_train',y_states_train');
    W12 = obj.W;
    Q12 = W12'*y_data_train;
  
    Result.W12 = W12;
    Result.M12 = M12;
    Result.G12 = G12;
    Result.Q12 = Q12;
    Result.Target = y_states_train;
           
    Target = (2-y_states_train);
    Result.Acc = sum( (Q12>0) == Target )/length(y_states_train);
        
end

