function [IND EiRatio L dmax dmin] =  CSPDimReduction(data_1,data_2)

    Lambda = 0.1;
    
    C10 = data_1 * data_1' / size(data_1,2);
    C20 = data_2 * data_2' / size(data_2,2);

    nchan = size(C10,1);

    %regularize covariances
    C1 = C10 + Lambda * trace(C10) * eye(nchan) / nchan;
    C2 = C20 + Lambda * trace(C20) * eye(nchan) / nchan;
    % do generalized eigenvalue decomp
    [V d] = eig(C1,C2);
    %get original channel index
    ind0 = 1:size(data_1,1);
    v1 = V(:,1)/max(abs(V(:,1)));
    v2 = V(:,end)/max(abs(V(:,end)));
    k = 1;
    CC1 = C1;
    CC2 = C2;
    for th =0:0.01:0.99
        ind = unique([find(abs(v1)>th);find(abs(v2)>th) ] )
        [v d] = eig(CC1(ind,ind),CC2(ind,ind));
        dmax(k) = d(end,end);
        dmin(k) = d(1,1);
        thr(k) = th;
        L(k) = length(ind);
        v1 = v(:,1)/max(abs(v(:,1)));
        v2 = v(:,end)/max(abs(v(:,end)));
        CC1 = CC1(ind,ind);
        CC2 = CC2(ind,ind);
        ind0 = ind0(ind);
        IND{k} = ind0;
        EiRatio(k) = dmax(k)/dmin(k);
        k = k+1;
    end;    
