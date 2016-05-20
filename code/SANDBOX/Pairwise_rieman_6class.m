clear all;
close all;
%load('C:\Work\BCI\bci-master\EXP_DATA\EXP_LSL32_new\bci_expresult_LSL32_first_2603_first_imag_T20_1.mat')
%load('C:\Work\BCI\bci-master\EXP_DATA\EXP_LSL32_new\bci_expresult_LSL32_first_2603_first_real_T20.mat')
load('C:\Work\BCI\bci-master\EXP_DATA\EXP_LSL32_new\bci_expresult_LSL32_first_2603_first_imag_T20_2.mat');
%load('C:\Work\BCI\bci-master\EXP_DATA\EXP_LSL32_new\bci_expresult_LSL32_first_0804_main_imag_T20_1.mat');

data_cur = data.data;
states_cur = states.data;

Fs = 1000;
L = size(data_cur,2);
NFFT = 2^nextpow2(L);


Fc_low =14;
Fc_high =12;

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

Nc = max(states_cur);

for i = 1:Nc
    data_state{i} = data_cur(:,states_cur == i);
    
end

% figure;
A = zeros(1,20);
for reg = 1:1 
    for i = 1:Nc
       for j = 1:Nc
           if(i==j)
               continue;
           end;
        disp(i);
        disp(j);  
        data_1 = data_state{i};
        data_2 = data_state{j};
        
        ind_train1 = 1:fix(size(data_1,2)/2);
        ind_train2 = 1:fix(size(data_2,2)/2);

        data_1_train = data_1(:,ind_train1);
        data_2_train = data_2(:,ind_train2);

         R  = RiemDistFitPairwise( [data_1_train data_2_train], ...
             [ones(1,size(data_1_train,2)) 2*ones(1,size(data_2_train,2)) ],500);
        
        
        C1 = data_1_train * data_1_train' / size(data_1_train,2);
        C2 = data_2_train * data_2_train' / size(data_2_train,2);

        nchan = size(C1,1);
        C1 = C1 + (0.01*reg+0.01) * trace(C1) * eye(nchan) / size(C1,1);
        C2 = C2 + (0.01*reg+0.01) * trace(C2) * eye(nchan) / size(C2,1);

        [V d] = eig(C1,C2);

        M = V(:,[1:2, end-1:end])';
        MM{i,j} = M;
        Y1 = M * data_1;
        Y2 = M * data_2;


         y_data_train = [Y1(:,ind_train1).^2, Y2(:,ind_train2).^2];
         y_states_train = [ones(1,length(ind_train1)), 2*ones(1,length(ind_train2))];

         for k=1:size(y_data_train,1)
             y_data_train(k,:) = conv(y_data_train(k,:),ones(1,500),'same');
         end;


         obj = train_shrinkage(y_data_train',y_states_train');
         W = obj.W;
         Q0 = W'*y_data_train(:,find(y_states_train==1));
         Q1 = W'*y_data_train(:,find(y_states_train==2));
         if(mean(Q0)<mean(Q1))
            W = -obj.W;
            disp('flip');
         end;
         
         % Q for state 1 has to be greater than Q for state 2 
         
        WW{i,j} = W;
        ind_test1 = (fix(size(Y1,2)/2)+1):size(Y1,2);
        ind_test2 = (fix(size(Y2,2)/2)+1):size(Y2,2);
        y_data_test = [Y1(:,ind_test1).^2, Y2(:,ind_test2).^2];
        y_states_test= [ones(1,length(ind_test1)), 2*ones(1,length(ind_test2))];

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
    end %j
    A(reg) =0;
    for i = 1:Nc
        for j = i+1:Nc
            plot(1-Results{i,j}.spec,Results{i,j}.sens,'r.-');
            y = Results{i,j}.sens;
            x = 1-Results{i,j}.spec;
            for n=1:length(x)-1
               A(reg) = A(reg)+ 0.5*(y(n)+y(n+1))*(x(n)-x(n+1));
            end;
            hold on;
        end;
    end;
end
AA = zeros(Nc,Nc);
figure
for i = 1:Nc
    for j = i+1:Nc
        plot(1-Results{i,j}.spec,Results{i,j}.sens,'r.-');
        y = Results{i,j}.sens;
        x = 1-Results{i,j}.spec;
        for n=1:length(x)-1
            AA(i,j) = AA(i,j)+0.5*(y(n)+y(n+1))*(x(n)-x(n+1));
        end;
        hold on;
    end;
end;

data_cur(:,states_cur == i);
for i = 1:Nc
   for j = 1:Nc
       if(i==j)
           continue;
       end;
       M = MM{i,j};
       Y = M * data_cur;
       Y = Y.^2;
       for k=1:size(Y,1)
           YC(k,:) = conv(Y(k,:),ones(1,500),'same');
       end;
    
       QQ(i,j,:) = WW{i,j}'*YC;
   end %i
end %j

    
     
col = ['b','g','m','c','k'];

figure
for k = 1:Nc
    subplot(3,2,k);
    ind = find(states_cur==k);
    plot(squeeze(sum(QQ(k,:,ind),2)),'r','LineWidth',2);
    hold on
    c = 1;
    for kk=1:Nc
        if(k~=kk)
            rng = 1:Nc;
            rng(k)=[];
            plot(squeeze(sum(QQ(kk,rng,ind),2)),col(c));
            c = c+1;
        end;
    end;
end;

            
  