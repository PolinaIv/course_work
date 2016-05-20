clear all;
close all;
%load('C:\Work\BCI\bci-master\EXP_DATA\EXP_LSL32_new\bci_expresult_LSL32_first_2603_first_imag_T20_1.mat')
%load('C:\Work\BCI\bci-master\EXP_DATA\EXP_LSL32_new\bci_expresult_LSL32_first_2603_first_real_T20.mat')
load('C:\Work\BCI\bci-master\EXP_DATA\EXP_LSL32_new\bci_expresult_LSL32_first_2603_first_imag_T20_2.mat');
%load('C:\Work\BCI\bci-master\EXP_DATA\EXP_LSL32_new\bci_expresult_LSL32_first_0804_main_imag_T20_1.mat');

data_cur = data.data;
states_cur = states.data;
useful_range = find(states_cur~=0);
data_cur = data_cur(:,useful_range);
states_cur = states_cur(:,useful_range);

Fs = 1000;
L = size(data_cur,2);
NFFT = 2^nextpow2(L);


Fc_low =25;
Fc_high = 14;

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
for i = 1:Nc
   for j = i+1:Nc
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
                y_data_train(k,:) = conv(y_data_train(k,:),ones(1,750),'same');
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
            % do not use the testing set to find the optimal params
%             ind_test1 = (fix(size(Y1,2)/2)+1):size(Y1,2);
%             ind_test2 = (fix(size(Y2,2)/2)+1):size(Y2,2);
%             y_data_test = [Y1(:,ind_test1).^2, Y2(:,ind_test2).^2];
%             y_states_test= [ones(1,length(ind_test1)), 2*ones(1,length(ind_test2))];
% 
%              for k=1:size(y_data_test,1)
%                  y_data_test(k,:) = conv(y_data_test(k,:),ones(1,750),'same');
%              end;

            Q0 = W'*y_data_train(:,find(y_states_train==1));
            Q1 = W'*y_data_train(:,find(y_states_train==2));
            th1 = mean(Q1);
            th0 = mean(Q0);
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
         end %reg
         [maxv,maxind] = max(A(ij,:));
         Results{i,j}.ROC = ResultsReg(maxind);
         Results{i,j}.M = MM{maxind};
         Results{i,j}.W = WW{maxind};
         ij = ij+1;
    end %j
end %i

%generate secondary feature set
data_cur_train = data_cur(:,1:fix(size(data_cur,2)/2));
states_cur_train = states_cur(:,1:fix(size(states_cur,2)/2));
ij = 1;
for i = 1:Nc
   for j = i+1:Nc
       M = Results{i,j}.M;
       W = Results{i,j}.W;
       Y = M*data_cur_train;
       for k=1:size(Y,1)
           Yc(k,:) = conv(Y(k,:).^2,ones(1,750),'same');
       end;
       QQ(ij,:) = W'*Yc;
       ij = ij+1;
   end
end; 
ind_nz = find(states_cur_train>0);
[Zfda,Wfda] = fda(QQ(:,ind_nz),states_cur_train(1,ind_nz)',3);
clear Mc Sc isqrtSc;
for c = 1:Nc
    ind_c = find(states_cur_train==c);
    Mc(:,c) = mean(Zfda(:,ind_c),2);
    Sc{c} = bsxfun(@minus,Zfda(:,ind_c),Mc(:,c))*bsxfun(@minus,Zfda(:,ind_c),Mc(:,c))'/(length(ind_c)-5);
    isqrtSc{c} = inv(sqrtm(Sc{c}));
end;

% test performance
data_cur_test = data_cur(:,fix(size(data_cur,2)/2)+1:end);
states_cur_test = states_cur(:,fix(size(states_cur,2)/2)+1:end);
c =1 ;
for c = 1:Nc
    MHD(c,:) = 0.5*log(det(Sc{c})) + sum((isqrtSc{c}*bsxfun(@minus, Zfda, Mc(:,c))).^2,1);
end;

[logPmax,indMax] = min(MHD,[],1);

H= hist3([states_cur_test(ind_nz);indMax]',[6,6]);

for i=1:size(H,1)
    Hn(i,:) = H(i,:)/sum(H(i,:));
end
figure
imagesc(Hn)
colorbar
mean(diag(Hn))
