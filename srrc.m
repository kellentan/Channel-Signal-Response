% srrc.m
function srrc = srrc(beta, n)
    if n == 0
        srrc = (4*beta + pi - beta*pi) / pi;
    else
        constant = 4 * beta / pi;
        numerator = cos((1+beta)*pi*n/4) + sin((1-beta)*pi*n/4) / (n*beta);
        denominator = 1 - (beta*n)^2;
        srrc = constant * numerator / denominator;
    end
end