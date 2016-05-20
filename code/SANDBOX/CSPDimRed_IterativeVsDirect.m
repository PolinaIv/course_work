a = load('C:\Work\BCI\bci-master\EXP_DATA\EXP_LSL32_new\top31_1.mat');
dt = load('ClassifyPairDataTrain.mat');
C10cv = dt.data_1_train*dt.data_1_train';
C1cv = C10cv + Lambda * trace(C10cv) * eye(nchan) / nchan;
C20cv = dt.data_2_train*dt.data_2_train';
C2cv = C20cv + Lambda * trace(C20cv) * eye(nchan) / nchan;

ind0 = 1:31;
v1 = V(:,1)/max(abs(V(:,1)));
v2 = V(:,end)/max(abs(V(:,end)));
k = 1;
CC1 = C1;
CC2 = C2;
for th =0:0.05:0.17
    ind = unique([find(abs(v1)>th);find(abs(v2)>th) ] )
    [v d] = eig(CC1(ind,ind),CC2(ind,ind));
    dmax(k) = d(end,end);
    dmin(k) = d(1,1);
    thr(k) = th;
    L(k) = length(ind);
    k = k+1;
    v1 = v(:,1)/max(abs(v(:,1)));
    v2 = v(:,end)/max(abs(v(:,end)));
    CC1 = CC1(ind,ind);
    CC2 = CC2(ind,ind);
    ind0 = ind0(ind);
end;    
a.chanlocs_vis(ind).labels


v1 = V(:,1)/max(abs(V(:,1)));
v2 = V(:,end)/max(abs(V(:,end)));
k = 1;
CC1 = C1;
CC2 = C2;
for th =0:0.05:0.2
    ind = unique([find(abs(v1)>th);find(abs(v2)>th) ] )
    [v d] = eig(CC1(ind,ind),CC2(ind,ind));
    dmax0(k) = d(end,end);
    dmin0(k) = d(1,1);
    thr0(k) = th;
    L0(k) = length(ind);
    k = k+1;
end;    
figure
[ax,p1,p2]=plotyy(thr0,L0,thr0,[dmin0; dmax0]');
hold on
[ax,p1,p2]=plotyy(thr,L,thr,[dmin; dmax]');



