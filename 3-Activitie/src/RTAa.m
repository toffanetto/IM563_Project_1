function [R,T,A]=RTAa(p_hat,dim)
% p_hat é a matriz de coords de imagem dos vértices da caixa
%  dim é vetor com as dimensões da caixa
%  A ordem dos valores em X deve obedecer a ordem dos pontos em VE
%  M, N são as quantidades
W=dim(1);
D=dim(2);
H=dim(3);
B=[ 0 W W W 0 0 0 W
    0 0 D D D 0 D 0
    0 0 0 H H H 0 H];
map=nchoosek(1:8,3);
P=zeros(length(map)*3,9);
for k=1:length(map)
    P_hat_I=[0.5*p_hat(:,map(k,1)) -p_hat(:,map(k,2)) 0.5*p_hat(:,map(k,3))];
    B_I=0.5*B(:,map(k,1))-B(:,map(k,2))+0.5*B(:,map(k,3));
    P_hat_II=[-1*p_hat(:,map(k,1)) 0.5*p_hat(:,map(k,2)) 0.5*p_hat(:,map(k,3))];
    B_II=-B(:,map(k,1))+0.5*B(:,map(k,2))+0.5*B(:,map(k,3));
    P((3*(k-1)+1):(3*(k-1)+3),:)=kron(B_I',eye(3))-kron(B_II',P_hat_I/P_hat_II);
end
[V,~]=eig(P'*P);
vP=V(:,1);
AR=reshape(vP,3,3);
AAt=AR*AR';
AAt=AAt/AAt(3,3);
u0=AAt(1,3);
v0=AAt(2,3);
Sxf=sqrt(AAt(1,1)-u0^2);
Syf=sqrt(AAt(2,2)-v0^2);

% A=[abm 0 N/2;0 abm M/2;0 0 1];

A=[Sxf 0 u0;0 Syf v0;0 0 1];
R=A\AR;

[U,~,V]=svd(R);
Rt=U*V';

MTR=kron(eye(8),ones(1,3))'.*kron(ones(8,1),p_hat);
MTR=[MTR, kron(ones(8,1),eye(3))];
VEC=A*Rt*B;
VEC=VEC(:);
LT=MTR\VEC;
Tt=(A*Rt)\LT(9:11);
if(mean(LT(1:8))<0)
    Tt=-Tt;
    Rt=-Rt;
end
R=Rt;
T=Tt;
end