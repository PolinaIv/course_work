clear all;
close all;
%load('C:\Work\BCI\bci-master\EXP_DATA\EXP_LSL32_new\bci_expresult_LSL32_first_2603_first_imag_T20_1.mat')
%load('C:\Work\BCI\bci-master\EXP_DATA\EXP_LSL32_new\bci_expresult_LSL32_first_2603_first_real_T20.mat')
%load('C:\Work\BCI\bci-master\EXP_DATA\EXP_LSL32_new\bci_expresult_LSL32_first_2603_first_imag_T20_2.mat');
%load('C:\Work\BCI\bci-master\EXP_DATA\EXP_LSL32_new\bci_expresult_LSL32_first_0804_main_imag_T20_1.mat');
%load('C:\Work\BCI\bci-master\EXP_DATA\EXP_LSL32_new\bci_expresult_LSL32_first_6states_2204_imag_');
%load('C:\Work\BCI\bci-master\EXP_DATA\EXP_LSL32_new\bci_expresult_LSL32_first_6states_3004_imag_12min.mat');
%EEGDummy = load_dataset('C:\Work\BCIClone\bci\EXP_DATA\EXP_LSL32_new\short_32chan_2.set');
% topoplot2(randn(32,1),EEGDummy.chanlocs,'electrodes','labelpoint','chaninfo',EEGDummy.chaninfo);
% topoplot(randn(32,1),EEGDummy.chanlocs,'electrodes','labelpoint','chaninfo',EEGDummy.chaninfo);
%load('C:\Work\BCI\bci-master\EXP_DATA\EXP_LSL32_new\firsttest.mat');
%load('C:\Work\BCI\bci-master\EXP_DATA\EXP_LSL32_new\bci_expresult_LSL32_first_2603_first_real_T20.mat');
load('G:\0705_alex_im_main_1.mat');
data_cur0 = resample(data.data',1,5)';
states_cur0 = states.data(1:5:end);

useful_range = find(states_cur0~=0);
data_cur0 = data_cur0(:,useful_range);
states_cur0 = states_cur0(:,useful_range);

data_pwr = sqrt(sum((data_cur0.^2),1));

 for n = 1 : 7
    Xmean = mean(data_pwr);
    Xstd = std(data_pwr);
    mask = (abs(data_pwr-Xmean) < 2.5 * Xstd);
    idx = find(mask);
    data_cur0 = data_cur0(:,idx);
    states_cur0 = states_cur0(idx);
    data_pwr = data_pwr(:,idx);
    length(idx)
 end
 
eye_art_ind = find(data_cur0(1,:) > 3*mean(abs(data_cur0(1,:))));

[u s v] = svd(data_cur0(:,eye_art_ind));
P = eye(size(u,1))-u(:,1:3)*u(:,1:3)';
data_cur0 = P*data_cur0;

HBW = 2;
Fs = 200;
h = 1;
l= 1;
STATES = [1,2,5,6];
for si=1:length(STATES)
    for sj=1:length(STATES)
        h = 1;
        if(si==sj) continue; end;
        for Fc1 = HBW+1:HBW:20
            B1 = Fc1-HBW:Fc1+HBW;
            [z_high,p_high,k_high] = butter(3, B1(1)/(Fs/2), 'high');
            [b_high1,a_high1] = zp2tf(z_high,p_high,k_high);

            [z_low,p_low,k_low] = butter(3, B1(end)/(Fs/2), 'low');
            [b_low1,a_low1] = zp2tf(z_low,p_low,k_low);
            l = 1;
            for Fc2 = HBW+1:HBW:20
                B2 = Fc2-HBW:Fc2+HBW;
                % skip if bands overlap > 50%
                if(length(intersect(B1,B2))>0.5*length(B1)) continue; end;
                
                [z_high,p_high,k_high] = butter(3, B2(1)/(Fs/2), 'high');
                [b_high2,a_high2] = zp2tf(z_high,p_high,k_high);

                [z_low,p_low,k_low] = butter(3, B2(end)/(Fs/2), 'low');
                [b_low2,a_low2] = zp2tf(z_low,p_low,k_low);
     
                % filter
                data_cur_h = filtfilt(b_high1, a_high1,data_cur0')'; 
                data_cur_hl = filtfilt(b_low1, a_low1,data_cur_h')'; 
                data_cur1 = data_cur_hl(:,1:2:end);
                data_cur_h = filtfilt(b_high2, a_high2,data_cur0')'; 
                data_cur_hl = filtfilt(b_low2, a_low2,data_cur_h')'; 
                data_cur2 = data_cur_hl(:,1:2:end);
                
                data_cur = {data_cur1, data_cur2};
                
                states_cur = states_cur0(:,1:2:end);

                T = size(data_cur,2);

                ind1 = find(states_cur ==STATES(si));
                ind2 = find(states_cur ==STATES(sj));

                indTr1 = ind1(1:fix(length(ind1)/2));
                indTr2 = ind2(1:fix(length(ind2)/2));

                indTst1 = ind1(fix(length(ind1)/2)+1:end);
                indTst2 = ind2(fix(length(ind2)/2)+1:end);

                for r=1:10
                    ResVsReg{r}= ClassifyPair2Bands(data_cur, indTr1,indTr2,0.05*r,200,2);
                    Acc(r) = ResVsReg{r}.Acc;
                end;
                
                [accMax, indMax] = max(Acc);
                
                ResultTest = TestPair2Bands(data_cur, indTst1,indTst2,ResVsReg{indMax}, 200,2);
                
                ResVsBand{h,l} = ResVsReg{indMax};
                ResVsBand{h,l}.AccTst = ResultTest.Acc;
                ResVsBand{h,l}.B1 = B1;
                ResVsBand{h,l}.B2 = B2;
                AccVsBand(h,l) = ResultTest.Acc;

                [si sj  h  l  ResultTest.Acc  accMax ]
                l = l+1;
            end; %Fc_high
            h = h+1;
        end; %Fc_low 
        [maxval, hlmax] = max(AccVsBand(:));
        [hmax,lmax] = ind2sub(size(AccVsBand),hlmax);
        RESULT{si,sj} = ResVsBand{hmax,lmax};
      end;
end;
%extract the out of sample accuracy 
for si=1:length(STATES)
    for sj=1:length(STATES)
        if(si==sj) continue; end;
        PWAcc(si,sj) = RESULT{si,sj}.AccTst;
    end;
end;
%draw and indicate bands and acc
figure
h = imagesc(PWAcc)
hold on;
for si=1:length(STATES)
    for sj=1:length(STATES)
        if(si==sj) continue; end;
        text(si-0.5,sj,sprintf('%2.2f - %2.2f Hz', RESULT{si,sj}.B1(1), RESULT{si,sj}.B1(end)));
        text(si-0.5,sj-0.25,sprintf('%2.2f - %2.2f Hz', RESULT{si,sj}.B2(1), RESULT{si,sj}.B2(end)));
        text(si,sj+0.25,sprintf('P=%2.2f', RESULT{si,sj}.AccTst));
    end;
end;
return;

%generate secondary feature set
D = size(RESULT{1,2}.M12,1);
rangeQQY = 1:D;
ij = 1;
for si=1:length(STATES)
    for sj=1:length(STATES)
 
        if(si==sj) continue; end;
        
        Fc_high = RESULT{si,sj}.Fc_high;
        Fc_low  = RESULT{si,sj}.Fc_low;
        [z_high,p_high,k_high] = butter(3, Fc_high/(Fs/2), 'high');
        [b_high,a_high] = zp2tf(z_high,p_high,k_high);
        [z_low,p_low,k_low] = butter(3, Fc_low/(Fs/2), 'low');
        [b_low,a_low] = zp2tf(z_low,p_low,k_low);

        % filter
        data_cur_h = filtfilt(b_high, a_high,data_cur0')'; 
        data_cur_hl = filtfilt(b_low, a_low,data_cur_h')'; 

        data_cur = data_cur_hl(:,1:2:end);
        states_cur = states_cur0(:,1:2:end);
        M = RESULT{si,sj}.M12;
        W = RESULT{si,sj}.W12;
        Y = M*data_cur;
        for k=1:size(Y,1)
           Yc(k,:) = conv(Y(k,:).^2,ones(1,200),'same');
        end;
        QQ(ij,:) = W'*Yc;
        QQY(rangeQQY,:) = Yc;
        rangeQQY = rangeQQY+D;
        ij = ij+1;
    end;
end;
indTr = 1:fix(length(states_cur)/2);

states_cur_train = states_cur(indTr);

QQ_train = QQ(:,indTr);
%QQY_train = QQY(:,indTr);

Nc = length(STATES);
target = zeros(Nc,size(states_cur,2));
for c = 1:Nc
    target(c,find(states_cur==STATES(c)))=1;
end;

target_train = target(:,indTr);
[u s v] = svd(QQY_train,'econ');

clear net2;
net2 = patternnet(10);
net2.divideFcn = 'divideind';
net2.divideParam.trainInd = indTr(1:2:end);
net2.divideParam.valInd = indTr(2:2:end);
net2.divideParam.testInd = indTr(end)+1:size(QQ,2);

net2.trainParam.min_grad = 1e-7;
net2.trainParam.max_fail = 30;
net2.trainParam.epochs = 5000;
[net1,tr] = train(net2,u(:,1:12)'*QQY,target);

Ztrain = net1(u(:,1:5)'*QQ);
figure
imagesc([Ztrain ; target]);

% test performance
data_cur_test = data_test;
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

QQtest = QQtest(GoodPairs,:);

%QQtest = QQtest(ind_good_pw,:);
Ztest = net1(QQtest);
figure
imagesc(Ztest)






return;


% topoplot(randn(32,1),EEGDummy.chanlocs,'electrodes','labelpoint','chaninfo',EEGDummy.chaninfo);a
a = load('g:\top31_1.mat');
figure
topoplot(a.Minv(:,1),EEGDummy.chanlocs,'electrodes','labelpoint','chaninfo',EEGDummy.chaninfo);a
