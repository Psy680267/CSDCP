function [U, V, Error] = CSDCP(fea, layers, gnd, NSelLoc, count, alpha, beta, maxiter )
%%% Correntropy based semi-supervised deep NMF with constraints propagation
num_of_layers = numel(layers);
U = cell(1, num_of_layers);
V = cell(1, num_of_layers);
[mFeat,~] = size(fea);
k = length(unique(gnd));
Error = zeros(1,maxiter); 
%%%%% construct informative weight matrix based on hypergraph based constraint propagation (HCP) algorithm %%%%%% 
p = 4; [H, W] = constructHW(fea',p); De = diag(sum(H,2)); Wx = H'*W*inv(De)*H;    
Wx = constructW_PC(gnd, NSelLoc, Wx, alpha, count); 
Wx = beta*Wx; DColx = full(sum(Wx,2)); Dx = spdiags(DColx,0,speye(size(Wx,1)));  Lx = Dx - Wx; 
%%%%%%%%%%% initial U and V %%%%%%%%%%%%%%%%%%%%  
for i_layer = 1:length(layers)  
    if i_layer == 1
        X = fea;
    else 
        X = V{i_layer-1};
    end  
    [U{i_layer}, V{i_layer}] = inital_CSDCP(X, layers(i_layer), Wx, Dx); 
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for iter = 1:maxiter  
    obj_Lap = 0;
    for i = 1:numel(layers)
    if i==1
        X = fea;     
    else
        X = V{i-1};
    end
%%%%%%%%%%%% update H %%%%%%%%%%%%
        if i == 1
           UU = U{i}; 
        else
           UU = UU*U{i}; 
        end                 
        kerwidth=sum(sum((fea-UU*V{i}).^2))/(2*mFeat);     
        diag_vector = exp(-sum((fea-UU*V{i}).^2,2)/(2*kerwidth))./kerwidth; 
        H = diag(diag_vector./diag_vector);   
%%%%%%%%%%%% update U %%%%%%%%%%%%
        if i == 1
            upper = H*X*V{i}';
            below = H*U{i}*V{i}*V{i}';
            U{i} = U{i}.*(upper./below);
            clear upper below;         
        else
            upper = K'*H*fea*V{i}';
            below = K'*H*K*U{i}*V{i}*V{i}';
            U{i} = U{i}.*(upper./max(below, 1e-6));
            clear upper below;
        end
%%%%%%%%%%%% update V %%%%%%%%%%%%                
        if i == 1
            K = U{1};
        else
            K = K*U{i};
        end
        upper = K'*H*fea+V{i}*Wx; 
        below = K'*K*V{i}+V{i}*Dx;           
        V{i} = V{i} .* (upper./max(below, 1e-6));  
        clear upper below;
        obj_Lap = obj_Lap + trace(V{i}*Lx*V{i}');
    end
    Error(iter) = cost_function(fea, U, V) + obj_Lap;
end
end

function error = cost_function(X, U, V)
    error = norm(X - reconstruction(U, V), 'fro');
    error = error.^2;
end

function [ out ] = reconstruction(U, V)
    out = V{numel(V)};
    for k = numel(V) : -1 : 1
        out =  U{k} * out;
    end
end
