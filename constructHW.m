function [H, W] = constructHW(fea, p)
% Each row of H consists of a hyper edge.
% Each column of H means the sample is used to consist of the corresponding
% hyper edge.

[nSmp, ~] = size(fea);
G = zeros(nSmp*(p+1),3);

dist = EuDist2(fea);
D = dist.^2;
sigma = mean(mean(dist));
%sigma=sqrt(2);
A = exp(D/(-sigma.^2));

dump = zeros(nSmp, p+1);
idx = dump;

for j = 1:p+1
    [dump(:,j),idx(:,j)] = min(dist,[],2);
    temp = (idx(:,j)-1)*nSmp+[1:nSmp]';
    dist(temp) = 1e100;
end

G(1:nSmp*(p+1),1) = repmat([1:nSmp]',[p+1,1]);
G(1:nSmp*(p+1),2) = idx(:);
G(1:nSmp*(p+1),3) = 1;

H = sparse(G(:,1),G(:,2),G(:,3),nSmp,nSmp);
W = diag(sum(A.*H'));
end

