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
%load('C:\Work\BCI\bci-master\EXP_DATA\EXP_LSL32_new\0705_alex_im_main_1.mat');
load('C:\Work\BCI\bci-master\EXP_DATA\EXP_LSL32_new\0705_alex_re_first.mat');

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
data_cur01 = P*data_cur0;


Fs = 200;
h = 1;
l= 1;
STATES = [1,2,5,6];
si = 1;
sj = 2;
Fc_high = 10;
Fc_low = 14;
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

ind1 = find(states_cur ==STATES(si));
ind2 = find(states_cur ==STATES(sj));

indTr1 = ind1(1:fix(length(ind1)/2));
indTr2 = ind2(1:fix(length(ind2)/2));

indTst1 = ind1(fix(length(ind1)/2)+1:end);
indTst2 = ind2(fix(length(ind2)/2)+1:end);
Res = ClassifyPair(data_cur, indTr1,indTr2,0.1,300,2);
EEGDummy = load_dataset('C:\Work\BCIClone\bci\EXP_DATA\EXP_LSL32_new\short_32chan_2.set');
a = load('C:\Work\BCI\bci-master\EXP_DATA\EXP_LSL32_new\top31_1.mat');
figure
for i=1:4
    subplot(2,2,i)
    topoplot(Res.G12(i,:),a.chanlocs_vis,'electrodes','labelpoint','chaninfo',EEGDummy.chaninfo);
end;

W = Res.M12(2,:);
W = W/norm(W);

yw = W*data_cur(:,indTr1);

y = data_cur(:,indTr1);
P = sum(y.^2,2);
[val ind] = max(P);
figure
plot(y(ind,:));
hold on;
plot(yw,'r')

