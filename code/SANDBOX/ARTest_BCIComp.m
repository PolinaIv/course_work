close all
clear all
bCov = true;
cd C:\Users\user\Downloads\BCICIV_1_mat
load BCICIV_calib_ds1a
cnt= 0.1*double(cnt)';

data_pwr = sqrt(sum((cnt.^2),1));
data_cur0 = cnt;
[b,a] = butter(3,2/50,'high');
data_cur0 = filtfilt(b,a,data_cur0);

Fs = 100;
h = 1;
l= 1;

states_cur = zeros(1,size(data_cur0,2));
for i= 1:length(mrk.pos)
    states_cur(mrk.pos(i):mrk.pos(i)+ 400-1) = mrk.y(i);
end;
Fc_high = 7;
Fc_low = 17;

[z_high,p_high,k_high] = butter(3, Fc_high/(Fs/2), 'high');
[b_high,a_high] = zp2tf(z_high,p_high,k_high);

[z_low,p_low,k_low] = butter(3, Fc_low/(Fs/2), 'low');
[b_low,a_low] = zp2tf(z_low,p_low,k_low);

% filter
data_cur_h  = filtfilt(b_high, a_high,data_cur0')'; 
data_cur    = filtfilt(b_low, a_low,data_cur_h')'; 

ind1 = find(states_cur ==1);
ind2 = find(states_cur ==0);

indTr1 = ind1(1:fix(length(ind1)/2));
indTr2 = ind2(1:fix(length(ind2)/2));

indTst1 = ind1(fix(length(ind1)/2)+1:end);
indTst2 = ind2(fix(length(ind2)/2)+1:end);

Result= ClassifyPairAR(data_cur, indTr1,indTr2,0.004*i,1);

