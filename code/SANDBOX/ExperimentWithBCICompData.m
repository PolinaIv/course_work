close all
clear all
cd C:\Users\user\Downloads\BCICIV_1_mat
load BCICIV_calib_ds1a
cnt= 0.1*double(cnt)';

data_pwr = sqrt(sum((cnt.^2),1));
data_cur = cnt;
[b,a] = butter(3,2/50,'high');
data_cur = filtfilt(b,a,data_cur);

target = zeros(1,size(data_cur,2));
for i= 1:length(mrk.pos)
    target(mrk.pos(i):mrk.pos(i)+ 400-1) = mrk.y(i);
end;


ind1 = find(target==1);
ind2 = find(target==-1);
indTr1 = ind1(1:end/2);
indTr2 = ind2(1:end/2);
indTst1 = ind1(end/2+1:end);
indTst2 = ind2(end/2+1:end);
data_1_train = data_cur(:,indTr1);
data_2_train = data_cur(:,indTr2);
    
C10 = data_1_train * data_1_train' / size(data_1_train,2);
C20 = data_2_train * data_2_train' / size(data_2_train,2);

nchan = size(C10,1);

%regularize covariances
C1 = C10 + 0.05 * trace(C10) * eye(nchan) / nchan;
C2 = C20 + 0.05 * trace(C20) * eye(nchan) / nchan;

% do generalized eigenvalue decomp
[V d] = eig(C1,C2);
iV = inv(V);
Ncomp = 5;
M12 = V(:,[1:Ncomp, end-Ncomp+1:end])';
Tsmooth = 200;
   Y1 = M12 * data_1_train;
    Y2 = M12 * data_2_train;

    y_data_train = [Y1.^2, Y2.^2];
    y_states_train = [ones(1,length(indTr1)), 2*ones(1,length(indTr2))];

    % compute average over time square(variance)
    for k=1:size(y_data_train,1)
        y_data_train(k,:) = conv(y_data_train(k,:),ones(1,Tsmooth),'same')/Tsmooth;
    end;
    
    % build shrinkage (linear) classifier
    obj = train_shrinkage(y_data_train',y_states_train');
    W12 = obj.W;
    Q12 = W12'*y_data_train;
figure
plot(Q12)
data_1_test= data_cur(:,indTst1);
data_2_test = data_cur(:,indTst2);
Y1 = M12 * data_1_test;
Y2 = M12 * data_2_test;

y_data_test = [Y1.^2, Y2.^2];
y_states_test = [ones(1,length(indTst1)), 2*ones(1,length(indTst2))];

% compute average over time square(variance)
for k=1:size(y_data_test,1)
    y_data_test(k,:) = conv(y_data_test(k,:),ones(1,Tsmooth),'same')/Tsmooth;
end;    

Q12test = W12'*y_data_test;
figure
plot(Q12test)

 for n = 1 : 7
    Xmean = mean(data_pwr);
    Xstd = std(data_pwr);
    mask = (abs(data_pwr-Xmean) < 2.5 * Xstd);
    idx = find(mask);
    data_cur = data_cur(:,idx);
 %   states_cur0 = states_cur0(idx);
    data_pwr = data_pwr(:,idx);
    length(idx)
 end 
 
 
target = zeros(1,size(data_cur,2));
for i= 1:length(mrk.pos)
    target(mrk.pos(i):mrk.pos(i)+ 400-1) = mrk.y(i);
end;

 OG = sum(cnt(1:10,:).^2,1);
