function [ data_cur, states_cur, data_state ] = PreprocessData( data_cur, states_cur,Fc_low, Fc_high,Fs)
    useful_range = find(states_cur~=0);
    data_cur = data_cur(:,useful_range);
    states_cur = states_cur(:,useful_range);

    L = size(data_cur,2);
  
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

    Nc = max(states_cur);

    for i = 1:Nc
        data_state{i} = data_cur(:,states_cur == i);

    end


end

