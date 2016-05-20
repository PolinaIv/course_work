clear all
loaded = load('C:\Work\BCI\bci-master\EXP_DATA\EXP_LSL32_new\bci_expresult_LSL32_first_2603_first_imag_T20_2.mat');
data = loaded.data.data;
states = loaded.states.data;

Fs = 1000;
L = size(data,2);
Nch = size(data,1);
clear Cr Dmax
%first scan frequency
Nc = max(states);
 clear Dr De;
for i=1:1
    for j=2:2
        ind_ij = find(states==i | states == j);
        states_cur = states(ind_ij);
        Cr = zeros(1,12);
        for bnd = 2:20
            Fc_low = 1+bnd*1;
            Fc_high = bnd*1-1;
            [z_high,p_high,k_high] = butter(5, Fc_high/(Fs/2), 'high');
            [b_high,a_high] = zp2tf(z_high,p_high,k_high);
            data_flt = filtfilt(b_high, a_high,data')'; 

            [z_low,p_low,k_low] = butter(5, Fc_low/(Fs/2), 'low');
            [b_low,a_low] = zp2tf(z_low,p_low,k_low);
            data_flt = filtfilt(b_low, a_low,data_flt')'; 
            data_cur = data_flt(:,ind_ij);
           
            borders = [1 find(abs(diff(states_cur))>=1) length(states_cur)];
            Cz1 = zeros(Nch);
            Cz2 = zeros(Nch);
            ind_i = find(states_cur==i);
            ind_j = find(states_cur==j);
            Cz1 = data_cur(:,ind_i)*data_cur(:,ind_i)'/length(ind_i);
            Cz2 = data_cur(:,ind_j)*data_cur(:,ind_j)'/length(ind_j);
            [V{bnd},D] = eig(Cz1 + 0.01*trace(Cz1)/Nch*eye(size(Cz1)),Cz2+ 0.01*trace(Cz2)/Nch*eye(Nch));
            P = V{bnd}(:,[1:3 end-2:end]);
             for b = 2:length(borders)
                range = borders(b-1):borders(b);
                x = P'*data_cur(:,range);
                st = fix(mean(states_cur(range))+0.01);
                st
                C{b-1}.C = x*x'/size(x,2);
                C{b-1}.S =st;
                SS(b-1) = st;
             end;
             [SS1 idx]=sort(SS);
             Cs = C(idx);
             for ii = 1:length(Cs)
               for jj=1:length(Cs)
                  Dst(ii,jj) = RiemDist(diag(diag(Cs{ii}.C)), diag(diag(Cs{jj}.C)));
               end;
            end;
            Cr(bnd) = (sum(sum(Dst(1:2,3:4))) + sum(sum(Dst(3:4,1:2))))/(sum(sum(Dst(1:2,1:2)))+ sum(sum(Dst(3:4,3:4))));
            Cr(bnd)
            bnd
        end
        [mxv mxind] = max(Cr); 
        P = V{mxind}(:,[1:3 end-2:end]);
        Fc_low = 1+mxind*2;
        Fc_high = mxind*2;
        [z_high,p_high,k_high] = butter(5, Fc_high/(Fs/2), 'high');
        [b_high,a_high] = zp2tf(z_high,p_high,k_high);
        data_flt = filtfilt(b_high, a_high,data')'; 

        [z_low,p_low,k_low] = butter(5, Fc_low/(Fs/2), 'low');
        [b_low,a_low] = zp2tf(z_low,p_low,k_low);
        data_flt = filtfilt(b_low, a_low,data_flt')'; 
        ind_ij = find(states==i | states == j);
        states_cur = states(ind_ij);
        data_cur = data_flt(:,ind_ij);
        borders = [1 find(abs(diff(states_cur))>=1) length(states_cur)];
         clear C SS
     
         for b = 2:length(borders)
            range = borders(b-1):borders(b);
            x = P'*data_cur(:,range);
            st = fix(mean(states_cur(range))+0.01);
            st
            C{b-1}.C = x*x'/size(x,2);
            C{b-1}.S =st;
            SS(b-1) = st;
          end;
          [SS1 idx]=sort(SS);
          Cs = C(idx);
          Dr{i,j}= zeros(length(Cs));          De{i,j} = zeros(length(Cs));
          for ii = 1:length(Cs)
                for jj=1:length(Cs)
                   Dr{i,j}(ii,jj) = RiemDist(diag(diag(Cs{ii}.C)), diag(diag(Cs{jj}.C)));
                   d = diag((Cs{ii}.C - Cs{jj}.C));
                    De{i,j}(ii,jj) = sum(d(:).^2);
                end;
          end;
    end;
end;

figure
for i=1:Nc-1
    subplot(2,3,i);
    imagesc(Dr{i,6}); 
    colorbar
end;
figure
for i=1:Nc-1
    subplot(2,3,i);
    imagesc(De{i,6}); 
    colorbar
end;
return
plot(Cr)
grid
CrMxInd = input('Enter peak index:')
%[CrMx CrMxInd] = max(Cr)
P = V{CrMxInd}(:,[1:3 end-2:end]);
Fc_low = 2+CrMxInd*2;
Fc_high = CrMxInd*2;
data_cur = data;
[z_high,p_high,k_high] = butter(5, Fc_high/(Fs/2), 'high');
[b_high,a_high] = zp2tf(z_high,p_high,k_high);
data_cur = filtfilt(b_high, a_high,data_cur')'; 
[z_low,p_low,k_low] = butter(5, Fc_low/(Fs/2), 'low');
[b_low,a_low] = zp2tf(z_low,p_low,k_low);
data_cur = filtfilt(b_low, a_low,data_cur')';

states_cur = states;
borders = [1 find(abs(diff(states))==1) length(states)];
clear C SS
k = zeros(1,2);
for b = 2:length(borders)
    range = borders(b-1):borders(b);
    x = P'*data_cur(:,range);
    st = fix(mean(states(range))+0.01);
    st
    k(st)=k(st)+1;
    C{b-1}.C = x*x'/size(x,2);
    C{b-1}.S =st;
    SS(b-1) = st;
end;
[SS1 idx]=sort(SS);
Cs = C(idx);
D = zeros(length(Cs));
De = zeros(length(Cs));
for i = 1:length(Cs)
    for j=1:length(Cs)
       D(i,j) = RiemDist(diag(diag(Cs{i}.C)), diag(diag(Cs{j}.C)));
       d = diag((Cs{i}.C - Cs{j}.C));
        De(i,j) = sum(d(:).^2);
    end;
end;
figure
subplot(1,2,1), imagesc(D)
subplot(1,2,2), imagesc(De)

