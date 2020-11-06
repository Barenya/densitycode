%   This is the simplified code for density weight function
%   Authors are requested to cite the two papers if they use this code in their work
%   1. Density weighted support vector machines for binary class imbalance learning and
%   2. Density weighted twin support vector machines for binary class imbalance learning
%   Here k is the number of k-nearest neighbor
function [s1] = weight(Data,k)
Data = [Data,Data];
[IDX, D] = knnsearch(Data, Data, 'K', k+1);  %%,  to find k+1 nearest neighbors to exclude the point to itself and then ignore the first.
IDX = IDX(:, 2:end);    %discard the point to itself
D = D(:, 2:end);   		%discard the distance of the point to itself
furthest_k_distance = D(:,end);
furthest_kth_distance1 =max(furthest_k_distance); %the number of closest neighbours eligible to be considered
s1=1-furthest_k_distance/furthest_kth_distance1; % resultant weights
end

