function [ ResultBest] = TestPairChooseComps(data, indTst1,indTst2,Res, Tsmooth,Ncomp)
      
    % calculate class covariances\
    data_1_test = data(:,indTst1);
    data_2_test = data(:,indTst2);
    
    for c=1:size(Res.M12,1)
        M12 = Res.M12(1:c);
        %get the power of the output
        Y1 = M12 * data_1_test;
        Y2 = M12 * data_2_test;

        y_data_test = [Y1.^2, Y2.^2];
        y_states_test = [ones(1,length(indTst1)), 2*ones(1,length(indTst2))];

        % compute average over time square(variance)
        for k=1:size(y_data_test,1)
            y_data_test(k,:) = conv(y_data_test(k,:),ones(1,Tsmooth),'same')/Tsmooth;
        end;

        % build shrinkage (linear) classifier
        W12 = Res.W12;
        Q12 = W12'*y_data_test;

        Target = (2-y_states_test);
        Result{c}.Acc = sum( (Q12>0) == Target )/length(y_states_test);
        Result{c}.Target = Target;
        Result{c}.Q12  = Q12;
        Result{c}.Input = y_data_test;
        Result{c}.Ncomp = c;
        Acc(c) = Result.Acc;
    end;
   
    [maxAcc,maxInd] = max(Acc);
    ResultBest = Result{maxInd};
end

