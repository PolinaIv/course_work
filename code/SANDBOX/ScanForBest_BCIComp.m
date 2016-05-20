close all
clear all
bCov = false;
cd C:\Users\user\Downloads\BCICIV_1_mat
load BCICIV_calib_ds1a
cnt= 0.1*double(cnt)';

data_pwr = sqrt(sum((cnt.^2),1));
data_cur0 = cnt;
[b,a] = butter(3,2/50,'high');
data_cur0 = filtfilt(b,a,data_cur0')';

Fs = 100;
h = 1;
l= 1;

states_cur = zeros(1,size(data_cur0,2));
for i= 1:length(mrk.pos)
    states_cur(mrk.pos(i):mrk.pos(i)+ 400-1) = mrk.y(i);
end;

for Fc_high = 2:18
    l = 1;

    for Bw = 3:5
        Fc_low = Fc_high+Bw;
        % make filters
        [z_high,p_high,k_high] = butter(3, Fc_high/(Fs/2), 'high');
        [b_high,a_high] = zp2tf(z_high,p_high,k_high);

        [z_low,p_low,k_low] = butter(3, Fc_low/(Fs/2), 'low');
        [b_low,a_low] = zp2tf(z_low,p_low,k_low);
        
        % filter
        data_cur_h = filtfilt(b_high, a_high,data_cur0')'; 
        data_cur = filtfilt(b_low, a_low,data_cur_h')'; 

               
        T = size(data_cur,2);
        
        ind1 = find(states_cur ==1);
        ind2 = find(states_cur ==-1);
        
        indTr1 = ind1(1:fix(length(ind1)/2));
        indTr2 = ind2(1:fix(length(ind2)/2));

        indTst1 = ind1(fix(length(ind1)/2)+1:end);
        indTst2 = ind2(fix(length(ind2)/2)+1:end);

        for i=1:10
            
            if(bCov)
                Result{i} = ClassifyPairCov(data_cur, indTr1,indTr2,0.004*i,200,50,5);
            else
                Result{i}= ClassifyPair(data_cur, indTr1,indTr2,0.004*i,200,5);
            end;
            
            Acc(i) = Result{i}.Acc;
        end;

        [accMax, indMax] = max(Acc);
        
        Res(h,l) = accMax;
        
        if(bCov)
           ResultTest = TestPairCov(data_cur, indTst1,indTst2,Result{indMax},200,50);
        else
            ResultTest = TestPair(data_cur, indTst1,indTst2,Result{indMax}, 200,5);
        end
        ResHL{h,l} = Result{indMax};
        
        ResTst(h,l) = ResultTest.Acc;
      
        [h l accMax ResultTest.Acc]
        l = l+1
    end;
    h = h+1
end;

figure
subplot(1,3,1)
imagesc(3:8,2:25, Res);
xlabel('W_1+');
ylabel('W_1');
title('Training');
colorbar
subplot(1,3,2)
imagesc(3:8,2:25, ResTst);
xlabel('W_1+');
ylabel('W_1');
title('Testing');
colorbar
subplot(1,3,3)
imagesc(3:8,2:25, Res-ResTst);
xlabel('W_1+');
ylabel('W_1');
title('Training-Testing');
colorbar
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
