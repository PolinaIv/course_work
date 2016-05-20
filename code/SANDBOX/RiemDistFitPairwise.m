function [ R ] = RiemDistFitPairwise( samples, classes,seg_length )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
class_nums = sort(unique(classes));
zer = find(class_nums==0);
class_nums(zer)=[];
Nc = length(class_nums);

%Find CSP subsapce\
if(length(class_nums)>2)
    disp('Only binary classification, so no more than two classes allowed');
    return;
end;

for c =1:Nc
    ind = find(classes==class_nums(c));
    C{c} = samples(:,ind)*samples(:,ind)'/length(ind);
    Nch = size(C{c},1);
    C{c} = C{c} + 0.01*trace(C{c})/Nch*eye(Nch);
end;

for c1 = 1:Nc
    for c2 = c1+1:Nc
        % find CSP based projector for pairwise constrast
        [V,E] = eig(C{c1},C{c2});
       [~, sort_key] = sort(diag(E));
        M{c1,c2} = V(:,[sort_key(1:2) sort_key(end-1:end)]);
        % calculate seed covariances for class c1 based on the M projected
        % data
        ind = find(classes==c1);
        samplesp = M{c1,c2}'*samples(:,ind);
        range = 1:seg_length;
        i = 1;
        while range(end)<size(samplesp,2)
            C1{c1,c2,i} = samplesp(:,range)*samplesp(:,range)'/seg_length;
            range = range+seg_length;
            i = i+1;
        end;
        % calculate seed covariances for class c2 based on the M projected
        % data
        ind = find(classes==c2);
        samplesp = M{c1,c2}'*samples(:,ind);
        range = 1:seg_length;
        i = 1;
        while range(end)<size(samplesp,2)
            C2{c1,c2,i} = samplesp(:,range)*samplesp(:,range)'/seg_length;
            range = range+seg_length;
            i = i+1;
        end;
        %Test within and across classes distances
        for i=1:size(C1,3)
            for j=1:size(C1,3)
                D11(i,j) = RiemDist(C1{c1,c2,i},C1{c1,c2,j});
            end;
        end;
        for i=1:size(C2,3)
           for j=1:size(C2,3)
               D22(i,j) = RiemDist(C2{c1,c2,i},C2{c1,c2,j});
           end;
        end;
        for i=1:size(C1,3)
           for j=1:size(C2,3)
               D12(i,j) = RiemDist(C2{c1,c2,i},C2{c1,c2,j});
           end;
        end;
    D{c1,c2} = [D11 D12 ; D12' D22];
    end;
end;

R.C1 = C1;
R.C2 = C2;
R.D = D;
