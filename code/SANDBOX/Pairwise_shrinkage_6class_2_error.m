clear all;
close all;
%load('C:\Work\BCI\bci-master\EXP_DATA\EXP_LSL32_new\bci_expresult_LSL32_first_2603_first_imag_T20_1.mat')
%load('C:\Work\BCI\bci-master\EXP_DATA\EXP_LSL32_new\bci_expresult_LSL32_first_2603_first_real_T20.mat')
%load('C:\Work\BCI\bci-master\EXP_DATA\EXP_LSL32_new\bci_expresult_LSL32_first_2603_first_imag_T20_2.mat');
%load('C:\Work\BCI\bci-master\EXP_DATA\EXP_LSL32_new\bci_expresult_LSL32_first_0804_main_imag_T20_1.mat');
load('C:\Work\BCI\bci-master\EXP_DATA\EXP_LSL32_new\bci_expresult_LSL32_first_6states_2204_imag_');

data_cur = data.data;
states_cur = states.data;
useful_range = find(states_cur~=0);
data_cur = data_cur(:,useful_range);
states_cur = states_cur(:,useful_range);
Tsmooth = 300;
Fs = 1000;
L = size(data_cur,2);
NFFT = 2^nextpow2(L);


Fc_low =35;
Fc_high = 7;

%Wn =  Fc /(Fs/2)

[z_high,p_high,k_high] = butter(3, Fc_high/(Fs/2), 'high');
[b_high,a_high] = zp2tf(z_high,p_high,k_high);
data_cur = filtfilt(b_high, a_high,data_cur')'; 

[z_low,p_low,k_low] = butter(3, Fc_low/(Fs/2), 'low');
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
        
        ind_train1 = 1:fix(size(data_1,2)/2);
        ind_train2 = 1:fix(size(data_2,2)/2);

        data_1_train = data_1(:,ind_train1);
        data_2_train = data_2(:,ind_train2);

        C1 = data_1_train * data_1_train' / size(data_1_train,2);
        C2 = data_2_train * data_2_train' / size(data_2_train,2);

        nchan = size(C1,1);
        A(ij,:) = zeros(1,20);
        for reg = 1:20 
            C1 = C1 + (0.01*reg+0.01) * trace(C1) * eye(nchan) / size(C1,1);
            C2 = C2 + (0.01*reg+0.01) * trace(C2) * eye(nchan) / size(C2,1);

            [V d] = eig(C1,C2);

            M = V(:,[1:4, end-3:end])';
            
            MM{reg} = M;
            
            Y1 = M * data_1;
            Y2 = M * data_2;
            y_data_train = [Y1(:,ind_train1).^2, Y2(:,ind_train2).^2];
            y_states_train = [ones(1,length(ind_train1)), 2*ones(1,length(ind_train2))];

            for k=1:size(y_data_train,1)
                y_data_train(k,:) = conv(y_data_train(k,:),ones(1,Tsmooth),'same');
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
            WW{reg} = W;
            
            ind_test1 = (fix(size(Y1,2)/2)+1):size(Y1,2);
            ind_test2 = (fix(size(Y2,2)/2)+1):size(Y2,2);
            y_data_test = [Y1(:,ind_test1).^2, Y2(:,ind_test2).^2];
            y_states_test= [ones(1,length(ind_test1)), 2*ones(1,length(ind_test2))];

             for k=1:size(y_data_test,1)
                 y_data_test(k,:) = conv(y_data_test(k,:),ones(1,Tsmooth),'same');
             end;
             
            Q0 = W'*y_data_test(:,find(y_states_test==1));
            Q1 = W'*y_data_test(:,find(y_states_test==2));
            th1 = min(Q1);
            th0 = max(Q0);
            % compute 20 points of ROC curve
            dth = (th1-th0)/100;
            k=1;
            for k=1:100
                th = th0+dth*(k-1);
                ResultsReg(reg).sens(k) = length(find(Q1<=th))/length(Q1);
                ResultsReg(reg).spec(k) = length(find(Q0>th))/length(Q0);
            end;
             y = ResultsReg(reg).sens;
             x = 1-ResultsReg(reg).spec;
             A(ij,reg) = 0;
             for n=1:length(x)-1
                A(ij,reg) = A(ij,reg)+ 0.5*(y(n)+y(n+1))*(x(n)-x(n+1));
             end;
             if(reg==5 & j==3 ) %A(ij,reg)>0.4)
                 goodf = 1;
             end;
         end %reg
         [maxv,maxind] = max(A(ij,:));
         Results{ij}.ROC = ResultsReg(maxind);
         Results{ij}.M = MM{maxind};
         Results{ij}.W = WW{maxind};
         Results{ij}.AUC = maxv;
         
         ij = ij+1;
    end %j
end %i

figure
for ij=1:length(Results)
    if(Results{ij}.AUC<0.7)
        plot(1-Results{ij}.ROC.spec,Results{ij}.ROC.sens,'b.-');
    else
        plot(1-Results{ij}.ROC.spec,Results{ij}.ROC.sens,'r.-');
    end;
    hold on;
end;

%generate secondary feature set
data_cur_train = data_cur;%(:,1:fix(size(data_cur,2)/2));
states_cur_train = states_cur;%(:,1:fix(size(states_cur,2)/2));

clear Yc Y QQ;
ij = 1;
for i = 1:Nc
   for j = 1:Nc
       if(i==j) continue; end;
       M = Results{ij}.M;
       W = Results{ij}.W;
       Y = M*data_cur_train;
       for k=1:size(Y,1)
           Yc(k,:) = conv(Y(k,:).^2,ones(1,Tsmooth),'same');
       end;
       QQ(ij,:) = W'*Yc;
       ij = ij+1;
   end

end;
%ind_good_pw = find(Amax>0.8);

target = zeros(Nc,size(states_cur_train,2));
for c = 1:Nc
    target(c,find(states_cur_train==c))=1;
end;

net1 = patternnet(10);
view(net1)
[net1,tr] = train(net1,QQ,target);
Ztrain  = net1(QQ);
figure
imagesc(Ztrain);

load('C:\Work\BCI\bci-master\EXP_DATA\EXP_LSL32_new\bci_expresult_LSL32_first_6states_2204_imag_2');

data_cur = data.data;
states_cur = states.data;
useful_range = find(states_cur~=0);

data_cur = data_cur(:,useful_range);
states_cur = states_cur(:,useful_range);

data_cur = filtfilt(b_high, a_high,data_cur')'; 
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

%%%%%%%%%%%%%%%%%%%%%
% test performance
data_cur_test = data_cur;
states_cur_test = states_cur;

clear Ytest Yctest QQtest
ij = 1;
for i = 1:Nc
   for j = 1:Nc
       if(i==j) continue; end;
       M = Results{ij}.M;
       W = Results{ij}.W;
       Ytest = M*data_cur_test;
       for k=1:size(Ytest,1)
           Yctest(k,:) = conv(Ytest(k,:).^2,ones(1,Tsmooth),'same');
       end;
       QQtest(ij,:) = W'*Yctest;
       ij = ij+1;
   end
end; 

Ztest  = net1(QQtest);
figure
[states_cur_test_srt key] = sort(states_cur_test);
imagesc(Ztest(:,key));

