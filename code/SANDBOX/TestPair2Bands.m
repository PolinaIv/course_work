function [ Result] = TestPair2Bands(datab, indTst1,indTst2,Res, Tsmooth,Ncomp)
  
    Nb = length(datab);
   
    for b = 1:Nb
        % calculate class covariances
        data_1_test{b} = datab{b}(:,indTst1);
        data_2_test{b} = datab{b}(:,indTst2);

        Y1{b} = Res.M12{b} * data_1_test{b};
        Y2{b} = Res.M12{b} * data_2_test{b};
    end
    
    %get the power of the output
    y_data_test = [];
    for b = 1:length(Y1)
         y_data_test = [y_data_test; [ Y1{b}.^2, Y2{b}.^2 ]];
    end;
    
    y_states_test = [ones(1,length(indTst1)), 2*ones(1,length(indTst2))];

    % compute average over time square(variance)
    for k=1:size(y_data_test,1)
        y_data_test(k,:) = conv(y_data_test(k,:),ones(1,Tsmooth),'same')/Tsmooth;
    end;
   
    % get linear classifier's weights
    W12 = Res.W12;
    Q12 = W12'*y_data_test;
     
    Target = (2-y_states_test);
    Result.Acc = sum( (Q12>0) == Target )/length(y_states_test);
    Result.Target = Target;
    Result.Q12  = Q12;
    Result.Input = y_data_test;
end

