clear all;
%close all;
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
load('C:\Work\BCI\bci-master\EXP_DATA\EXP_LSL32_new\0705_alex_im_main_1.mat');
%load('C:\Work\BCI\bci-master\EXP_DATA\EXP_LSL32_new\0705_alex_re_first.mat');
%load('C:\Work\BCI\bci-master\EXP_DATA\EXP_LSL32_new\29_05_lesha_im_first.mat');
%load('C:\Work\BCI\bci-master\EXP_DATA\EXP_LSL32_new\1305_lisa_re_first.mat');
bDoSSD = true;
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

Fs = 200;
[u s v] = svd(data_cur0(:,eye_art_ind));
P = eye(size(u,1))-u(:,1:3)*u(:,1:3)';
data_cur01 = P*data_cur0;
data_cur00 = data_cur0;
ChCnt = 1;
for kkk = 5:30
    
if(bDoSSD)
    k = 1;
    for i = 4:25
        Fc = i;
       [de_srt, v_srt, topo_srt] = ssd(data_cur00,Fs, [Fc-2.5 Fc-0.5; Fc-0.5 Fc+0.5; Fc+0.5 Fc+2.5]);
    %    [de_srt, v_srt, topo_srt] = ssd(data_cur01,Fs, [Fc-0.5 Fc+0.5]);
        H(k) = de_srt(1);
        F(k) = Fc;
        V{k}= v_srt;
        Tp{k} = topo_srt;
        k = k+1
    end;
    [UU SS VV] = svd([V{6}(:,1:3) V{8}(:,1:3) ]);
    [crit key_srt] = (sort(sum(abs(UU(:,1:6)),2)));
    data_cur0 = data_cur00(key_srt(end-kkk:end),:);
end;
%data_cur0 = data_cur0( [4    10    15    17    19    25    27],:);
%data_cur0 = data_cur0( [17    20    22    23    27],:);
%data_cur0 = data_cur0([ 9    15    17    20    26    27    30],:);

Fs = 200;
h = 1;
l= 1;
bDoReduction = false;
STATES = [1,2,5,6];
for si=1:length(STATES)
    for sj=1:length(STATES)
        h = 1;
        if(si==sj) continue; end;
        for Fc_high = 8:2:14
            l = 1;
            for Bw = 3:6
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

                ind1 = find(states_cur ==STATES(si));
                ind2 = find(states_cur ==STATES(sj));

                indTr1 = ind1(1:fix(length(ind1)/2));
                indTr2 = ind2(1:fix(length(ind2)/2));

                indTst1 = ind1(fix(length(ind1)/2)+1:end);
                indTst2 = ind2(fix(length(ind2)/2)+1:end);
                
                ResVsNcmp= ClassifyPair(data_cur, indTr1,indTr2,0.05,200,2);
                ResultTest = TestPair(data_cur, indTst1,indTst2,ResVsNcmp, 200);
                
                if(bDoReduction)
                    [IND EiRatio L dmax dmin] =  CSPDimReduction(data_cur(:,indTr1),data_cur(:,indTr2));
                end;
                
                ResVsBand{h,l} = ResVsNcmp;
                ResVsBand{h,l}.AccTst = ResultTest.Acc;
                ResVsBand{h,l}.Fc_high = Fc_high;
                ResVsBand{h,l}.Fc_low = Fc_low;
                if(bDoReduction)
                    ResVsBand{h,l}.RedSet  = IND;
                    ResVsBand{h,l}.EiRatio  = EiRatio;
                    ResVsBand{h,l}.ReducedCnt = L;
                end;
                AccVsBand(h,l) = ResultTest.Acc;

                [si sj  h  l  ResultTest.Acc ]
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
PWAcc1 = PWAcc;
PWAcc1(find(PWAcc1==0))=[];
Perf(ChCnt,:) = [kkk+1 min(PWAcc1(:)) median(PWAcc1(:)) max(PWAcc1(:))];
ChCnt = ChCnt+1;
end
return
%draw and indicate bands and acc
figure
h = imagesc(PWAcc)
hold on;
for si=1:length(STATES)
    for sj=1:length(STATES)
        if(si==sj) continue; end;
        text(si-0.5,sj,sprintf('%2.2f - %2.2f Hz', RESULT{si,sj}.Fc_high, RESULT{si,sj}.Fc_low));
        text(si,sj+0.25,sprintf('P=%2.2f', RESULT{si,sj}.AccTst));
    end;
end;
if(bDoReduction)
    ElHst = zeros(1,31);
    for si=1:length(STATES)
        for sj=1:length(STATES)
            if(si==sj) continue; end;
            ii = find(RESULT{si,sj}.ReducedCnt==4);
            if(isempty(ii))
                ii = find(RESULT{si,sj}.ReducedCnt==3);
            end
            if(isempty(ii))
                ii = find(RESULT{si,sj}.ReducedCnt==2);
            end
            ElHst(RESULT{si,sj}.RedSet{ii(1)}) = ElHst(RESULT{si,sj}.RedSet{ii(1)}) +1;
        end;
    end;
    a = load('C:\Work\BCI\bci-master\EXP_DATA\EXP_LSL32_new\top31_1.mat');
    for i=1: length(a.chanlocs_vis)
        ElNames{i} =  a.chanlocs_vis(i).labels;
    end;
    figure
    h = bar(ElHst);
    set(get(h,'Parent'),'XTick',1:length(ElNames));
    set(get(h,'Parent'),'XTickLabel',ElNames);
    axis tight
    EEGDummy = load_dataset('C:\Work\BCIClone\bci\EXP_DATA\EXP_LSL32_new\short_32chan_2.set');
    figure
    topoplot(ElHst,a.chanlocs_vis,'electrodes','labelpoint','chaninfo',EEGDummy.chaninfo);
end
 %a.chanlocs_vis([4    10    15    17    19    25    27]).labels
return
a = load('C:\Work\BCI\bci-master\EXP_DATA\EXP_LSL32_new\top31_1.mat');
SelChanIndices = key_srt(end-7:end);
save PWAccSSD4x4 PWAcc a SelChanIndices;
SelChanIndices = [4    10    15    17    19    25    27];
save PWAccCSP4x4 PWAcc a SelChanIndices;
a.chanlocs_vis(SelChanIndices).labels
return

% EEGDummy = load_dataset('C:\Work\BCIClone\bci\EXP_DATA\EXP_LSL32_new\short_32chan_2.set');
% a = load('C:\Work\BCI\bci-master\EXP_DATA\EXP_LSL32_new\top31_1.mat');
% figure
% for i=1:4
%     subplot(2,2,i)
% topoplot(RESULT{3,2}.G12(i,:),a.chanlocs_vis,'electrodes','labelpoint','chaninfo',EEGDummy.chaninfo);
% title(num2str(i))
% 
% end;
%X  = [5 0.51 0.88 0.66; 6 0.51 0.082 0.68; 7 0.64 0.82 0.75; 8 ]

