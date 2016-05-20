clear all;
close all;
%load('C:\Work\BCI\bci-master\EXP_DATA\EXP_LSL32_new\bci_expresult_LSL32_first_2603_first_imag_T20_1.mat')
%load('C:\Work\BCI\bci-master\EXP_DATA\EXP_LSL32_new\bci_expresult_LSL32_first_2603_first_real_T20.mat')
%load('C:\Work\BCI\bci-master\EXP_DATA\EXP_LSL32_new\bci_expresult_LSL32_first_2603_first_imag_T20_2.mat');
%load('C:\Work\BCI\bci-master\EXP_DATA\EXP_LSL32_new\bci_expresult_LSL32_first_0804_main_imag_T20_1.mat');
load('C:\Work\BCI\bci-master\EXP_DATA\EXP_LSL32_new\bci_expresult_LSL32_first_6states_2204_imag_');
%load('C:\Work\BCI\bci-master\EXP_DATA\EXP_LSL32_new\bci_expresult_LSL32_first_6states_3004_imag_12min.mat');


data_cur = resample(data.data',1,10)';
states_cur = states.data(1:10:end);

useful_range = find(states_cur~=0);
data_cur = data_cur(:,useful_range);
states_cur = states_cur(:,useful_range);

StepRange = 20;
Dur = 200;
Ncomp = 5;
Fs = 100;
L = size(data_cur,2);
NFFT = 2^nextpow2(L);


Fc_low =17
Fc_high = 7;

%Wn =  Fc /(Fs/2)

[z_high,p_high,k_high] = butter(3, Fc_high/(Fs/2), 'high');
[b_high,a_high] = zp2tf(z_high,p_high,k_high);
data_cur = filtfilt(b_high, a_high,data_cur')'; 

[z_low,p_low,k_low] = butter(3, Fc_low/(Fs/2), 'low');
[b_low,a_low] = zp2tf(z_low,p_low,k_low);
data_cur = filtfilt(b_low, a_low,data_cur')'; 
data_pwr = sqrt(sum((data_cur.^2),1));

 for n = 1 : 5
    Xmean = mean(data_pwr);
    Xstd = std(data_pwr);
    mask = (abs(data_pwr-Xmean) < 2.5 * Xstd);
    idx = find(mask);
    data_cur = data_cur(:,idx);
    states_cur = states_cur(idx);
    data_pwr = data_pwr(:,idx);
    length(idx)
 end

ind1 = find(states_cur ==1);
ind2 = find(states_cur ==2);

indTr1 = ind1(1:fix(length(ind1)/2));
indTr2 = ind2(1:fix(length(ind2)/2));

indTst1 = ind1(fix(length(ind1)/2)+1:end);
indTst2 = ind2(fix(length(ind2)/2)+1:end);

Result= ClassifyPairAR(data_cur, indTr1,indTr2,0.05,1);

