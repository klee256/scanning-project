function a=shuffle(a,shufl_amount)
    if nargin == 1
        shufl_amount=1;
    end

    for i=1:shufl_amount
        a=a(randperm(length(a)));
    end
 end