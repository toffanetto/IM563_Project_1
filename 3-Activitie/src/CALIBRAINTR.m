clear all
[p_hat,p_hat_teorico,R,t,A,f,B] = gera8;
% p_hat = p_hat_teorico;
W=0.087;
H=0.275;
D=0.150;
dim = [W;D;H];
% p_hat = p_hat_teorico;
[R_est,t_est,A_est] = RTAa(p_hat,dim)

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


function [p_hat,p_hat_teorico,R,t,A,f,B] = gera8
N = 8; %Número de pontos
W=0.2;% X direction
H=0.3;% Z direction
D=0.4;% Y direction
B=[ 0 W W W 0 0 0 W
    0 0 D D D 0 D 0
    0 0 0 H H H 0 H];

f = 0.01; %Distância focal da câmera
Lx = 0.005; %Dimensão do plano de imagem
Ly = 3/4*Lx; %Dimensão do plano de imagem

%Definindo matriz de rotação da câmera 
theta = [-90+10 -5 10]*pi/180;%ãngulos de rotação da câmera com respeito ao mundo
R_x = [1 0 0; 0 cos(theta(1)) sin(theta(1));0 -sin(theta(1)) cos(theta(1))];  
R_y = [cos(theta(2))  0 -sin(theta(2));0 1 0; sin(theta(2)) 0  cos(theta(2))];  
R_z = [cos(theta(3)) sin(theta(3)) 0; -sin(theta(3)) cos(theta(3)) 0; 0 0 1];
R = (R_z*R_y*R_x);
%Definindo vetor de translação da câmera com respeito a  caixa 
t = [1 ;-5; -1];
%Definindo as projecões espaciais de todos os pontos no referencial da câmera 
P = R*(B-t*ones(1,N));
%Definindo as projeções perspectivas (ou coordenadas de câmera)
p = f* P./(ones(3,1)*P(3,:));
%Definindo as projeções de imagem (pixels)
N_px = 984;  %número de pixels na direção horizontal
N_py = 3/4*N_px;  %número de pixels na direção vertical
u = round(N_px/2);
v = round(N_py/2);

s_x = N_px/Lx; % Densidade de pixels em x
s_y = N_py/Ly; % Densidade de pixels em y
A = [ s_x*f   0    u;
        0   s_y*f  v;
        0     0    1]; %Matriz de parêmtros intrínsecos das câmeras
p_hat_teorico = 1/f*A*p;   %Vetor homogêneo de coordenadas de imagem da câmera 
p_hat = round(p_hat_teorico);
% OBS: Na prática, os vetores de coordenadas de imagem são compostos de
% números inteiros, e não fracionários.  

%IMPRESSAO DOS PONTOS
 %M<ostra o cubo no espaço, e suas projeções perspectivas nas câmeras
axle_x = ([0 W;0 0;0 0]'*2-ones(2,1)*t')*R';
axle_y = ([0 0;0 D;0 0]'*2-ones(2,1)*t')*R';
axle_z = ([0 0;0 0;0 H]'*2-ones(2,1)*t')*R';
axcam_x = [0 Lx;0 0;0 0]';
axcam_y = [0 0;0 Ly;0 0]';
axcam_z = [0 0;0 0;0 5*f]';
implane = [Lx/2 Lx/2 -Lx/2 -Lx/2 Lx/2;Ly/2 -Ly/2 -Ly/2 Ly/2 Ly/2; f f f f f]';
B_pl = (B(:,[1:6 1 7 3 4 8 6 8 2 3 7 5])'-ones(17,1)*t')*R';
B_pl = B_pl./(B_pl(:,3)*ones(1,3))*f;

myGreen=[0 0.5 0];
figure(1)
% plot(axle_x(:,1),axle_x(:,2),'r',...
% axle_y(:,1),axle_y(:,2),'g',...
% axle_z(:,1),axle_z(:,2),'b',...
% axcam_x(:,1),axcam_x(:,2),'r',...
% axcam_y(:,1),axcam_y(:,2),'g',...
% axcam_z(:,1),axcam_z(:,2),'b',...
% implane(:,1),implane(:,2),'m',...
% B_pl(:,1),B_pl(:,2),'k',B_pl(:,1),B_pl(:,2),'+k');
% axis ([-Lx/2 Lx/2 -Lx/2 Lx/2]*1.1)
% set(gca,"yDir",'reverse')
img1 = [Lx/2 Lx/2 -Lx/2 -Lx/2 Lx/2 ; -Ly/2 Ly/2 Ly/2 -Ly/2 -Ly/2 ; f f f f f ]; 
ax1 = [0 0 0 0 0 0 f f 0;2 0 0 0 2 0 f f 20 ];
ax2 = R*reshape(ax1,6,3)'+t*ones(1,6);
ipl = [1 2 3 4 8 6  1 7 3 2 8];
% ax2=reshape(ax2,2,9);
% img2 = R*img1+t*ones(1,11);  
plot(p(1,:)',p(2,:)','.r',p(1,ipl)',p(2,ipl)','k',...
    img1(1,:)',img1(2,:)','r',...
    [p(1,1),p(1,2)],[p(2,1),p(2,2)],'r',...
    [p(1,1),p(1,7)],[p(2,1),p(2,7)],'g',...
    [p(1,1),p(1,6)],[p(2,1),p(2,6)],'b',...
    ax1(:,1),ax1(:,4),'r',ax1(:,2),ax1(:,5),'g',ax1(:,3),ax1(:,6),'b')
axis(1.1*[-Lx/2 Lx/2 -Ly/2 Ly/2])
set(gca,"yDir",'reverse')
title('Pontos projetados na câmera')
%  
%  img1 = [-Lx/2 -Lx/2 Lx/2 Lx/2 -Lx/2 0 -Lx/2 0 Lx/2 0 Lx/2; -Ly/2 Ly/2 Ly/2 -Ly/2 -Ly/2 0 Ly/2 0 Ly/2 0 -Ly/2; f f f f f 0 f 0 f 0 f]; 
% ax1 = [0 0 0 0 0 0 f f 0;2 0 0 0 2 0 f f 20 ];
% ax2 = R*reshape(ax1,6,3)'+t*ones(1,6);
% ax2=reshape(ax2,2,9);
% img2 = R*img1+t*ones(1,11); 
% % figure(1),plot3(P(1,:)',P(2,:)',P(3,:)','k+',img1(1,:)',img1(2,:)',img1(3,:)','b',img2(1,:)',img2(2,:)',img2(3,:)',...
% %     p(:,1),p(:,2),p(:,3),'b.',q_(:,1),q_(:,2),q_(:,3),'r.')
% figure(1),subplot(211),plot3(P(1,:)',P(3,:)',P(2,:)','k+',...
%     P(1,1),P(3,1),P(2,1),'ok',...
%     [P(1,1),P(1,2)],[P(3,1),P(3,2)],[P(2,1),P(2,2)],'r',...
%     [P(1,1),P(1,6)],[P(3,1),P(3,6)],[P(2,1),P(2,6)],'g',...
%     [P(1,1),P(1,7)],[P(3,1),P(3,7)],[P(2,1),P(2,7)],'b',...
%     img1(1,:)',img1(3,:)',img1(2,:)','r',...
%     ax1(:,1),ax1(:,7),ax1(:,4),'r',ax1(:,2),ax1(:,8),ax1(:,5),'g',ax1(:,3),ax1(:,9),ax1(:,6),'b',...
%     p(1,:)',p(3,:)',p(2,:)','r.')
% xlabel('X')
% ylabel('Z')
% zlabel('Y')
% title('Pontos no espaço')
% set(gca,"zDir",'reverse')
% % axis([-2 2 -2 40 -1 20])
% view(30,15)
% subplot(212),plot(p(1,:)',p(2,:)','.r',img1(1,:)',img1(2,:)','r',...
%     [p(1,1),p(1,2)],[p(2,1),p(2,2)],'r',...
%     [p(1,1),p(1,6)],[p(2,1),p(2,6)],'g',...
%     [p(1,1),p(1,7)],[p(2,1),p(2,7)],'b',...
%     ax1(:,1),ax1(:,4),'r',ax1(:,2),ax1(:,5),'g',ax1(:,3),ax1(:,6),'b')
% axis(1.1*[-Lx/2 Lx/2 -Ly/2 Ly/2])
% set(gca,"yDir",'reverse')
% title('Pontos projetados na câmera')
end




    

