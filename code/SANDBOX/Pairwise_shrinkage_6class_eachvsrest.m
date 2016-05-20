clear all;

load('C:\Work\BCI\bci-master\EXP_DATA\EXP_LSL32_new\bci_expresult_LSL32_first_2603_first_imag_T20.mat')
%load('C:\Work\BCI\bci-master\EXP_DATA\EXP_LSL32_new\bci_expresult_LSL32_first_2603_first_real_T20.mat') 
data_cur = data.data;
states_cur = states.data;

Fs = 1000;
L = size(data_cur,2);
NFFT = 2^nextpow2(L);


Fc_low =40;
Fc_high = 2;

%Wn =  Fc /(Fs/2)

[z_high,p_high,k_high] = butter(5, Fc_high/(Fs/2), 'high');
[b_high,a_high] = zp2tf(z_high,p_high,k_high);
data_cur = filtfilt(b_high, a_high,data_cur')'; 

[z_low,p_low,k_low] = butter(5, Fc_low/(Fs/2), 'low');
[b_low,a_low] = zp2tf(z_low,p_low,k_low);
data_cur = filtfilt(b_low, a_low,data_cur')'; 
data_cur = data_cur(:,1:2:end);
states_cur = states_cur(:,1:2:end);

data_pwr = sqrt(sum((data_cur.^2),1));

 for n = 1 : 1
    Xmean = mean(data_pwr);
    Xstd = std(data_pwr);
    mask = (abs(data_pwr-Xmean) < 2.5 * Xstd);
    mask = prod(double(mask),1);
    idx = find(mask);
    data_cur = data_cur(:,idx);
    states_cur = states_cur(idx);
    data_pwr = data_pwr(:,idx);
 end



for i = 1:6
    data_state{i} = data_cur(:,states_cur == i);
    
end

% figure;
A = zeros(1,20);
reg = 7;

for i = 1:6
    data_1 = data_state{i};
    ind_train1 = 1:3*fix(size(data_1,2)/4);
    data_1_train = data_1(:,ind_train1);
    C1 = data_1_train * data_1_train' / size(data_1_train,2);
    nchan = size(C1,1);
    C1 = C1 + (0.01*reg+0.01) * trace(C1) * eye(nchan) / size(C1,1);
    ind_test1 = 3*fix(size(data_1,2)/4)+1:size(data_1,2);
    data_1_test = data_1(:,ind_test1);
        
    C2 = zeros(nchan);
    Sz = 0;
    data_2_test = [];
    data_2_train = [];    
    for j = 1:6
        if(i~=j)
            data_2 = data_state{j};
            ind_train2 = 1:3*fix(size(data_2,2)/4);
            data_2_train =[data_2_train data_state{j}(:,ind_train2)];
            ind_test2 = 3*fix(size(data_2,2)/4)+1:size(data_2,2);
            data_2_test = [data_2_test  data_state{j}(:,ind_test2)];
        end;
    end;
    C2 = data_2_train * data_2_train'/size(data_2_train,2);
    
    C2 = C2 + (0.01*reg+0.01) * trace(C2) * eye(nchan) / size(C2,1);

    [V d] = eig(C1,C2);

    M = V(:,[1:3, end-2:end])';

    y_data_train = [(M * data_1_train).^2, (M * data_2_train).^2];
    y_states_train = [ones(1,length(ind_train1)), 2*ones(1,length(ind_train2))];

    for k=1:size(y_data_train,1)
        y_data_train(k,:) = conv(y_data_train(k,:),ones(1,500),'same');
    end;

    obj = train_shrinkage(y_data_train',y_states_train');
    W = obj.W;
    
    y_data_test = [(M * data_1_test).^2, (M * data_2_test).^2];
    y_states_test= [ones(1,size(data_1_test,2)), 2*ones(1,size(data_2_test,2))];

     for k=1:size(y_data_test,1)
         y_data_test(k,:) = conv(y_data_test(k,:),ones(1,500),'same');
     end;

    Q0 = W'*y_data_test(:,find(y_states_test==1));
    Q1 = W'*y_data_test(:,find(y_states_test==2));
    th1 = mean(Q1);
    th0 = mean(Q0);
    % compute 20 points of ROC curve
    dth = (th1-th0)/100;
    k=1;
    for k=1:100
        th = th0+dth*(k-1);
        Results{i,j}.sens(k) = length(find(Q1<=th))/length(Q1);
        Results{i,j}.spec(k) = length(find(Q0>th))/length(Q0);
    end;
end %i


A(reg) =0;
for i = 1:6
    for j = i+1:6
        plot(1-Results{i,j}.spec,Results{i,j}.sens,'r.-');
        y = Results{i,j}.sens;
        x = 1-Results{i,j}.spec;
        for n=1:length(x)-1
           A(reg) = A(reg)+ 0.5*(y(n)+y(n+1))*(x(n)-x(n+1));
        end;
        hold on;
    end;
end;

figure
plot(A)
AA = zeros(6,6);
figure
for i = 1:6
    for j = i+1:6
        plot(1-Results{i,j}.spec,Results{i,j}.sens,'r.-');
        y = Results{i,j}.sens;
        x = 1-Results{i,j}.spec;
        for n=1:length(x)-1
            AA(i,j) = AA(i,j)+0.5*(y(n)+y(n+1))*(x(n)-x(n+1));
        end;
        hold on;
    end;
end;

     
figure
imagesc(AA)
colorbar
[h1,a1] = hist(Q0,100);
[h2,a2] = hist(Q1,100);
figure
plot(a1,h1); hold on;
plot(a2,h2,'r')

