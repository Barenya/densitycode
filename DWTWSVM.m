%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   This is the simplified code for density weighted twin SVM method for nonlinear case using QPP from toolbox
% 	Should work on MATLAB 2016b and newer versions 
%   This code is made freely available for reusability purpose
%   Authors are requested to cite the two papers if they use this code in their work
%   1. Density weighted support vector machines for binary class imbalance learning and
%   2. Density weighted twin support vector machines for binary class imbalance learning
%   If anybody face diffculty while implementing this code, feel free to
%   contact us. 
%   Created by Barenya Bikash Hazarika and Deepak Gupta
%   email id: barenya1431@gmail.com
%   Two optimization problems are solved.i.e., QPP1 and QPP2
%   Consider,
%			 C = train data,
%	 test_data = test data
%			c1 = model parameter,  
%			mu = kernel parameter,
%			K  = k-nearest neighbour value;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [recall, precision, obs1,classifier, time]=DWTWSVM(C,test_data,c1,k,mu)
  %%%%%%%DENSITY WEIGHTS ASSIGNMENT%%%%%
  S = weight(C,k);
  [no_input,no_col]=size(C);
  obs = C(:,no_col);  
  c3=c1;
  P = [];
  Q = [];
  S1 = [];
  S2 = [];
  c4=c3;
for i = 1:no_input
    if(obs(i) == 1)
        P = [P;C(i,1:no_col-1)];
        S1 = [S1;S(i,:)];
    else
        Q = [Q;C(i,1:no_col-1)];
        S2 = [S2;S(i,:)];
    end
end
	D1=S1;
	D2=S2;

    [m1,n] = size(P); 
    e1 = ones(m1,1); %one's vector
    [m2,n] = size(Q);
    e2 = ones(m2,1);%one's vector
    m= m1 + m2;
    C = [P; Q];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    c2=c1;
    ep = 0.00001;
      
    tic
    K=zeros(m1,m);
     
    for i=1:m1
        for j=1:m
            nom = norm( P(i,:)  - C(j,:)  );
            K(i,j) = exp( -1/(2*mu*mu) * nom * nom );
        end
    end
       
     H = [K e1];
         
    
     K=zeros(m2,m);
    for i=1:m2
        for j=1:m
            nom = norm( Q(i,:)  - C(j,:)  );
            K(i,j) = exp( -1/(2*mu*mu) * nom * nom );
        end
    end

    G = [K e2];

    em1 = m+1;

    
    lowb1=zeros(m2,1);%LOWER BOUND
    lowb2=zeros(m1,1);%LOWER BOUND
%%%Density weights assignment%%%%%%%%%%%  
    upb1 = c1*D2;%UPPER BOUND
    upb2 = c2*D1;%UPPER BOUND
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
%     G = [K e];
    HTH = H' * H;
    invHTH = inv(HTH + ep * speye(em1) );
    GINVGT = G * invHTH * G';
    GTG = G' * G;
    invGTG = inv (GTG + ep * speye(em1));
    HINVHT = H * invGTG * H';

       f1 = -e2';
       f2 = -e1';
    
    u1 = quadprog(GINVGT,f1,[],[],[],[],lowb1,upb1); %%%QPP1
    u2 = quadprog(HINVHT,f2,[],[],[],[],lowb2,upb2); %%%QPP2
    time= toc
    %I=speye(size(invHTH));
     w1 = - invHTH* G' *u1;
     w2 =  invGTG * H' *u2;

    [no_test,no_col] = size(test_data);   
    Ker_row = zeros( no_test, m ); 
	%%%%%%%%%%%%%GAUSSIAN KERNEL%%%%%%%%%%%%%%%
     for i=1:no_test
        for j=1:m
            nom = norm( test_data(i,1:no_col-1)  - C(j,:)  );
            Ker_row(i,j) = exp( -1/(2*mu*mu) * nom * nom );
            
        end
     end
     K = [Ker_row ones(no_test,1)];
     y1 = K * w1 / norm(w1(1:size(w1,1)-1,:));
     y2 = K * w2 / norm(w2(1:size(w2,1)-1,:));
     
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


    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    