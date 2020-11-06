%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   This is the simplified code for density weighted least squares twin SVM method for nonlinear case 
% 	Should work on MATLAB 2016b and newer versions 
%   Authors are requested to cite the two papers if they use this code in their work
%   1. Density weighted support vector machines for binary class imbalance learning and
%   2. Density weighted twin support vector machines for binary class imbalance learning
%   If anybody face diffculty while implementing this code, feel free to
%   contact us. 
%   Created by Barenya Bikash Hazarika and Deepak Gupta
%   email id: barenya1431@gmail.com
%   Two linear programming problems (LPP) are solved.
%   Consider,
%			 C = train data,
%	 test_data = test data,
%			C1 = model parameter,  
%			mu = kernel parameter,
%			K  = k-nearest neighbour value.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [recall, precision, obs1,classifier, time]=DWLSTWSVM(C,test_data,C1,mu,K)
 [no_input,no_col]=size(C);
 obs = C(:,no_col);    
 A = [];
 B = [];

for i = 1:no_input
    if(obs(i) == 1)
        A = [A;C(i,1:no_col-1)];
    else
        B = [B;C(i,1:no_col-1)];
    end
end

     [m1,n] = size(A); 
     e1 = ones(m1,1); %one's vector
     [m2,n] = size(B);
     e2 = ones(m2,1); %one's vector
     m= m1 + m2;
     C = [A ; B];
	%%%Density weights assignment%%%%%%%%%%% 
     D1=diag(weight(A,K));
     D2=diag(weight(B,K));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
	C2=C1;
    ep = 0.00001;
      
    tic %%%%%%time calculation starts here
    K=zeros(m1,m);
	%%%%%%%%%Gaussian kernel%%%%%%%%%%% 
    for i=1:m1
        for j=1:m
            nom = norm( A(i,:)  - C(j,:)  );
            K(i,j) = exp( -1/(2*mu*mu) * nom * nom );
        end
    end
       
    G = [K e1];
            
    K=zeros(m2,m);
	%%%%%%%%%Gaussian kernel%%%%%%%%%%% 
    for i=1:m2
        for j=1:m
            nom = norm( B(i,:)  - C(j,:)  );
            K(i,j) = exp( -1/(2*mu*mu) * nom * nom );
        end
    end

    H = [K e2];
    
    em1 = m+1;
    
    u1 = -inv((G'*G/C1) + H'*D2*D2'*H + 0.01*speye(em1))*H'*D2*D2'*e2; %%%%LPP1
    u2 = inv((H'*H/C2) + G'*D1*D1'*G + 0.01*speye(em1))*G'*D1*D1'*e1;  %%%%LPP2

    time = toc;
	%%%%%%%%%time calculation ends here
    
    [no_test,no_col] = size(test_data);   
    Ker_row = zeros( no_test, m );
	%%%%%%%%%Gaussian kernel%%%%%%%%%%% 
     for i=1:no_test
        for j=1:m
            nom = norm( test_data(i,1:no_col-1)  - C(j,:)  );
            Ker_row(i,j) = exp( -1/(2*mu*mu) * nom * nom );
        end
     end
     K = [Ker_row ones(no_test,1)];
     y1 = K * u1 / norm(u1(1:size(u1,1)-1,:));
     y2 = K * u2 / norm(u2(1:size(u2,1)-1,:));
     
    for i = 1 : no_test
        if abs(y1(i)) < abs(y2(i))
            classifier(i) = 1;
        else
            classifier(i) = -1;
        end
    end
%-----------------------------
match = 0.;
classifier = classifier';
obs1 = test_data(:,no_col);
confmat=confusionmat(obs1,classifier,'order',[1,-1]);%%%confusion matrix
TP=confmat(1,1); %true positive
TN=confmat(2,2); %true negative
FP=confmat(2,1); %false positive
FN=confmat(1,2); %false negative
recall1=TP/(TP+FN);
precision1=TP/(TP+FP);
recall=recall1';
precision=precision1';