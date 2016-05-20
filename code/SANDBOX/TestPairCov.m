function [ Result] = TestPairCov(data, indTst1,indTst2,Res,Dur,StepRange)
      
    % calculate class covariances\
    data_1_test = data(:,indTst1);
    data_2_test = data(:,indTst2);
    
    M12 = Res.M12;
  
    Y1 = M12 * data_1_test;
    Y2 = M12 * data_2_test;
    
    LTIndex = [];
    
    iijj = 0;
    for ii=1:size(Y1,1)
        for jj=1:size(Y1,1)
            iijj = iijj+1;
            if(ii>=jj)
                LTIndex = [LTIndex iijj];
            end;
        end;
    end;
            
    %%%%%%%%%%%%
    range = 1:Dur;
    t = 1;
    while(range(end)<size(Y1,2))
         R = Y1(:,range)*Y1(:,range)'/length(range);
         R1(:,t) = R(LTIndex);
         range = range+StepRange;
         t = t+1;
    end;

    range = 1:Dur;
    t = 1;
    while(range(end)<size(Y2,2))
         R = Y2(:,range)*Y2(:,range)'/length(range);
         R2(:,t) = R(LTIndex);
         range = range+StepRange;
         t = t+1;
    end;

    y_data_test = [R1 R2];
    y_states_test = [ones(1,size(R1,2)),2*ones(1,size(R2,2))];

    W12 = Res.W12;
    Q12 = W12'*y_data_test;
  
   
    Result.Q12 = Q12;
    Result.Target = y_states_test;
           
    Target = (2-y_states_test);
    Result.Acc = sum( (Q12>0) == Target )/length(y_states_test);
        
end

