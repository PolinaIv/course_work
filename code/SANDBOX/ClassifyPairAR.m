function [ Result] = ClassifyPairAR(data, indTr1,indTr2,Lambda,Ncomp)
      
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
  
    AROrder = 3;
    Dim = 2;

    for ord = 1:AROrder 
        AL{ord} = logical(ones(Dim));
    end;

    Mdl1 = vgxset('ARsolve',AL);

    [EstSpec1,EstSE,logL,W] = vgxvarx(Mdl1,Y1');
    [EstSpec2,EstSE,logL,W] = vgxvarx(Mdl1,Y2');

    Y = [Y1 Y2];
    
    Yh1 = zeros(size(Y));
    Yh2 = zeros(size(Y));
    
    for t = AROrder+1:size(Y,2)
        val = zeros(Dim,1);
        for k=1:AROrder
            val = val + EstSpec1.AR{k}*Y(:,t-k);
        end;
        Yh1(:,t) = val;

        val = zeros(Dim,1);
        for k=1:AROrder
            val = val + EstSpec2.AR{k}*Y(:,t-k);
        end;
        Yh2(:,t) = val;
    end;
    
    Result.ARModel = EstSpec1;
    Result.ARSE = EstSE;
    Result.M12 = M12;
    Result.Yh1 = Yh1;
    Result.Yh2 = Yh2;
end

