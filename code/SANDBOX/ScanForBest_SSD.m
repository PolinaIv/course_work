clear all;
close all;
%load('C:\Work\BCI\bci-master\EXP_DATA\EXP_LSL32_new\bci_expresult_LSL32_first_2603_first_imag_T20_1.mat')
%load('C:\Work\BCI\bci-master\EXP_DATA\EXP_LSL32_new\bci_expresult_LSL32_first_2603_first_real_T20.mat')
%load('C:\Work\BCI\bci-master\EXP_DATA\EXP_LSL32_new\bci_expresult_LSL32_first_2603_first_imag_T20_2.mat');
load('C:\Work\BCI\bci-master\EXP_DATA\EXP_LSL32_new\bci_expresult_LSL32_first_0804_main_imag_T20_1.mat');
%load('C:\Work\BCI\bci-master\EXP_DATA\EXP_LSL32_new\bci_expresult_LSL32_first_6states_2204_imag_');
%load('C:\Work\BCI\bci-master\EXP_DATA\EXP_LSL32_new\bci_expresult_LSL32_first_6states_3004_imag_12min.mat');
load('C:\Work\BCI\bci-master\EXP_DATA\EXP_LSL32_new\firsttest.mat');

%EEGDummy = load_dataset('C:\Work\BCIClone\bci\EXP_DATA\EXP_LSL32_new\short_32chan_2.set');
% topoplot2(randn(32,1),EEGDummy.chanlocs,'electrodes','labelpoint','chaninfo',EEGDummy.chaninfo);
% topoplot(randn(32,1),EEGDummy.chanlocs,'electrodes','labelpoint','chaninfo',EEGDummy.chaninfo);
data_cur0 = resample(data.data',1,5)';
states_cur0 = states.data(1:5:end);

useful_range = find(states_cur0~=0);
data_cur0 = data_cur0(:,useful_range);
states_cur0 = states_cur0(:,useful_range);

%12 min dataset channel 27 - bad

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
Fs = 200;
% do SSD first

i = 1;
clear C1 SSD1 F1 V1;
f0_ind = 1;
for f0 = 8:0.5:20
    p(1,:) = [f0-1.5,f0-0.5];
    p(3,:) = [f0+0.5,f0+1.5];
    p(2,:) = [f0-0.5,f0+0.5];
    for f = 1:3
        [b{i}(f,:),a{i}(f,:)] = butter(5,p(f,:)/(0.5*Fs));
        x = filtfilt(b{i}(f,:),a{i}(f,:),double(data_cur0'))';
        C1{f} = x*x'/size(x,2);
        C1{f} = C1{f}+ 0.05*trace(C1{f})/size(C1{f},1)*eye(size(C1{f}));
    end;
    [v e] = eig(C1{2},0.5*(C1{1}+C1{3}));
    [mxv, mxi] = max(diag(e)); 
    SSD1(i) = mxv;
    V1(:,i) = v(:,mxi);
    iv = inv(v);
    G1(:,i) = iv(mxi,:)';
    F1(i) = f0;
    i = i+1
end;
figure
plot(F1,SSD1)
figure
EEGDummy = load_dataset('C:\Work\BCIClone\bci\EXP_DATA\EXP_LSL32_new\short_32chan_2.set');
topoplot(G1(:,7),EEGDummy.chanlocs,'electrodes','labelpoint','chaninfo',EEGDummy.chaninfo);


h = 1;
l= 1;

for Fc_high = 2:25
    l = 1;
    for Bw = 3:8
        Fc_low = Fc_high+Bw;
        % make filters
        [z_high,p_high,k_high] = butter(3, Fc_high/(Fs/2), 'high');
        [b_high,a_high] = zp2tf(z_high,p_high,k_high);

        [z_low,p_low,k_low] = butter(3, Fc_low/(Fs/2), 'low');
        [b_low,a_low] = zp2tf(z_low,p_low,k_low);
        
        % filter
        data_cur_h = filtfilt(b_high, a_high,data_cur0')'; 
        data_cur_hl = filtfilt(b_low, a_low,data_cur_h')'; 

        data_cur = data_cur_hl(:,1:2:end);
        states_cur = states_cur0(:,1:2:end);
        
        T = size(data_cur,2);
        
        ind1 = find(states_cur ==1);
        ind2 = find(states_cur ==2);
        
        indTr1 = ind1(1:fix(length(ind1)/2));
        indTr2 = ind2(1:fix(length(ind2)/2));

        indTst1 = ind1(fix(length(ind1)/2)+1:end);
        indTst2 = ind2(fix(length(ind2)/2)+1:end);

        for i=1:10
            Result{i}= ClassifyPair(data_cur, indTr1,indTr2,0.04*i,200,1);
            Acc(i) = Result{i}.Acc;
        end;

        [accMax, indMax] = max(Acc);
        Res(h,l) = accMax;
        ResultTest = TestPair(data_cur, indTst1,indTst2,Result{indMax}, 200,1);
     
        ResHL{h,l} = Result{indMax};
        
        ResTst(h,l) = ResultTest.Acc;
      
        [h l accMax ResultTest.Acc]
        l = l+1
    end;
    h = h+1
end;

figure
imagesc(3:8,2:25, ResTst);
xlabel('W_1+');
ylabel('W_1');
return
Fc_high = 3:25;
Fc_high+3:35

Fc_high = 24; Fc_low = 28;
   % make filters
[z_high,p_high,k_high] = butter(3, Fc_high/(Fs/2), 'high');
[b_high,a_high] = zp2tf(z_high,p_high,k_high);

[z_low,p_low,k_low] = butter(3, Fc_low/(Fs/2), 'low');
[b_low,a_low] = zp2tf(z_low,p_low,k_low);

% filter
data_cur_h = filtfilt(b_high, a_high,data_cur0')'; 
data_cur_hl = filtfilt(b_low, a_low,data_cur_h')'; 

data_cur = data_cur_hl(:,1:2:end);
states_cur = states_cur0(:,1:2:end);

T = size(data_cur,2);
indTr1 = find(states_cur ==6);
indTr2 = setdiff(1:T,indTr1);

for i=1:10
    Res1{i}= ClassifyPair(data_cur, indTr1,indTr2,0.004*i,200,4);
    Acc(i) = Res1{i}.Acc;
end;
figure
plot(Acc)
figure
EEGDummy = load_dataset('C:\Work\BCIClone\bci\EXP_DATA\EXP_LSL32_new\short_32chan_2.set');
for i=1:size( ResHL{23,2}.G12,1) 
    topoplot( ResHL{23,2}.G12(i,:),EEGDummy.chanlocs,'electrodes','labelpoint','chaninfo',EEGDummy.chaninfo);
    pause;
    hold off;
end;

% topoplot(randn(32,1),EEGDummy.chanlocs,'electrodes','labelpoint','chaninfo',EEGDummy.chaninfo);
