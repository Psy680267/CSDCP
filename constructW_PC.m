function W_New = constructW_PC(gnd, NSelLoc, W, alpha, count)
%
%%%%% L, W, D %%%%%%
Nsam = length(gnd);
W = (W+W')./2;
DCol = full(sum(W,2));
D = spdiags(DCol,0,speye(size(W,1)));
D = full(D);
L = D^(-1/2)*W*D^(-1/2); %%%%%%%  
%%%%%% construct Z with different percent for must-link and cannot-link constratints
Z = zeros(Nsam,Nsam);
for i = 1:count
    a = NSelLoc(i);    
    for j = i+1:count
        b = NSelLoc(j);
        if gnd(a)==gnd(b)
           Z(a,b)=1;
        else
           Z(a,b)=-1; 
        end
    end
end
Z = (Z+Z');
%%%%%%%%%%%%%%%%%%%%%%%%%
Fv = zeros(Nsam,Nsam);
Fh = zeros(Nsam,Nsam);
t=0;
while t<30
    Fv = alpha*L*Fv + (1-alpha)*Z;
    t = t+1;
end
t=0;
while t<30
    Fh = alpha*Fh*L + (1-alpha)*Fv;  
    t = t+1;
end
%%%%%%%%% construct a new weight matrix %%%%%%%%%%%%%%%%
W_New = zeros(Nsam,Nsam);
for i = 1:Nsam
    for j = i+1:Nsam
        if Fh(i,j)<0
           W_New(i,j)=(1+Fh(i,j))*W(i,j);
        else
           W_New(i,j)=1-(1-Fh(i,j))*(1-W(i,j)); 
        end
    end
end
W_New = (W_New+W_New');




