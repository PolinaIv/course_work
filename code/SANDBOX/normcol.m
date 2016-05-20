function [ y ] =normcol( x )

    y = x*diag(1./sqrt(sum(x.*x,1)));

end

