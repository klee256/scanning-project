
% ind_split, takes in a number n which represents the number of elements in
% some array. The user then puts how many sets they want that array split into and what
% percentage each set will carry by inputing decimal or fraction values that add up to
% one. Example: ind_split(10,0.2,0.8) this will create two sets, one with
% 20% of the values and another with 80% (2 and 8 respectively)


function split_sets = ind_split(n,varargin)

base = 1:1:n; base=shuffle(base,100);
numSets=nargin - 1;

% error check

sumInput=sum(cell2mat(varargin));

if sumInput ~= 1
    error('Values do not add to one')
elseif sumInput == 1
    if sum(floor(n*cell2mat(varargin))) == n
        split_sets = mat2cell(base,1,n*cell2mat(varargin));
    else
        remainder=rem(n,sum(floor(n*cell2mat(varargin))));
        if remainder ~= 1
            error('Rounding error')
        end
        hold_remainder = base(1:remainder);
        base = base(remainder+1:end);
        split_sets = mat2cell(base,1,floor(n*cell2mat(varargin)));
        split_sets{randi(numSets)}(end+length(remainder)) = hold_remainder;
          
    end
else
    error('Something went wrong')
end


