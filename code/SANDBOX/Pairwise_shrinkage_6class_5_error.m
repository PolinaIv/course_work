clear all;
close all;
%load('C:\Work\BCI\bci-master\EXP_DATA\EXP_LSL32_new\bci_expresult_LSL32_first_2603_first_imag_T20_1.mat')
%load('C:\Work\BCI\bci-master\EXP_DATA\EXP_LSL32_new\bci_expresult_LSL32_first_2603_first_real_T20.mat')
%load('C:\Work\BCI\bci-master\EXP_DATA\EXP_LSL32_new\bci_expresult_LSL32_first_2603_first_imag_T20_2.mat');
%load('C:\Work\BCI\bci-master\EXP_DATA\EXP_LSL32_new\bci_expresult_LSL32_first_0804_main_imag_T20_1.mat');
load('C:\Work\BCI\bci-master\EXP_DATA\EXP_LSL32_new\bci_expresult_LSL32_first_6states_2204_imag_');


data_cur0 = resample(data.data',1,10)';
states_cur0 = states.data(1:10:end);

StepRange = 20;
Dur = 100;
Ncomp = 5;
Fc_low =45
Fc_high = 5;
Fs = 100;
TrainingFraction = 0.5;
Whiten=false;

[data_cur, states_cur, data_state] = PreprocessData( data_cur0, states_cur0,Fc_low, Fc_high,Fs);
L1 = fix(size(data_state{6},2)/2);

if(Whiten)
    R6 = data_state{6}(:,1:L1)*data_state{6}(:,1:L1)'/L1;
    iR6sqrt = inv(sqrtm(R6));
    Nc = length(data_state);
    for i=1:Nc
        data_state{i} = iR6sqrt*data_state{i};
    end;
end

[Results,A] = PairwiseClassification4(data_state,TrainingFraction,Dur,StepRange,Ncomp); 
PairwiseClassification4(data_state,TrainingFraction,Dur,StepRange,Ncomp); 
figure, AUC =  PlotPWResults( Results, 0.7);
sum(AUC(:))
P = zeros(6,6);
for i=1:6
    for j=1:6 
        if(i==j) continue; end;
        th = 0.5*(mean(Results{i,j}.Qtrn1)+mean(Results{i,j}.Qtrn2));
        P(i,j) = (sum(Results{i,j}.Qtst1>th)+sum(Results{i,j}.Qtst2<th)) /(length(Results{i,j}.Qtst2) + length(Results{i,j}.Qtst1));
    end;
end;

    

net1 = patternnet(20);
view(net1)
[net1,tr] = train(net1,QQ,target);
Ztrain = net1(QQ);
% test performance
data_cur_test = data_test;
states_cur_test = states_test;

clear Ytest Yctest QQtest R
ij = 1;
for i = 1:Nc
   for j = 1:Nc
       if(i==j) continue; end;
       M = Results{ij}.M;
       W = Results{ij}.W;
       Y = M*data_cur_test;
       range = 1:Dur;
       t = 1;
       while(range(end)<size(Y,2))
            R = Y(:,range)*Y(:,range)'/length(range);
            RR(:,t) = R(LTIndex);
            range = range+StepRange;
            t = t+1;
       end;
       QQtest(ij,:)  = W'*RR;
       ij = ij+1;
   end;
end
targettest = zeros(Nc,size(QQtest,2));
range = 1:Dur;
t = 1;
while(range(end)<size(Y,2))
    h = hist(states_cur_test(range),[1:Nc]);
    h = h/sum(h);
    [maxv,maxind] = max(h);
    targettest(:,t) = zeros(Nc,1); targettest(maxind,t)=1; 
    range = range+StepRange;
    t = t+1;
end;




%QQtest = QQtest(ind_good_pw,:);
Ztest = net1(QQtest);
figure
subplot(2,1,1)
imagesc(Ztrain)
subplot(2,1,2)
imagesc(Ztest)

