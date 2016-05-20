function [ Result] = ClassifyPair2Bands(datab, indTr1,indTr2,Lambda,Tsmooth,Ncomp)
    
    Nb = length(datab);
   
    for b = 1:Nb
        % calculate class covariances
        data_1_train{b} = datab{b}(:,indTr1);
        data_2_train{b} = datab{b}(:,indTr2);

        C10 = data_1_train{b} * data_1_train{b}' / size(data_1_train{b},2);
        C20 = data_2_train{b} * data_2_train{b}' / size(data_2_train{b},2);

        nchan = size(C10,1);
    
        %regularize covariances
        C1 = C10 + Lambda * trace(C10) * eye(nchan) / nchan;
        C2 = C20 + Lambda * trace(C20) * eye(nchan) / nchan;
        % do generalized eigenvalue decomp
        [V d] = eig(C1,C2);
        iV = inv(V);
        M12{b} = V(:,[1:Ncomp, end-Ncomp+1:end])';
        G12{b} = iV([1:Ncomp, end-Ncomp+1:end],:);
        Y1{b} = M12{b} * data_1_train{b};
        Y2{b} = M12{b} * data_2_train{b};
    end
    
    %get the power of the output
    y_data_train = [];
    for b = 1:length(Y1)
         y_data_train = [y_data_train; [ Y1{b}.^2, Y2{b}.^2 ]];
    end;
    y_states_train = [ones(1,length(indTr1)), 2*ones(1,length(indTr2))];

    % compute average over time square(variance)
    for k=1:size(y_data_train,1)
        y_data_train(k,:) = conv(y_data_train(k,:),ones(1,Tsmooth),'same')/Tsmooth;
    end;
    
    % build shrinkage (linear) classifier
    obj = train_shrinkage(y_data_train',y_states_train');
    W12 = obj.W;
    Q12 = W12'*y_data_train;
 
    Result.W12 = W12;
    Result.M12 = M12;
    Result.G12 = G12;
    Result.Q12 = Q12;
    Result.Target = y_states_train;
    Result.Input  = y_data_train;
           
    Target = (2-y_states_train);
    Result.Acc = sum( (Q12>0) == Target )/length(y_states_train);
        
end

