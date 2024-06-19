%% Intrinsic Camera Calibration with a Box
% Image size: 640 x 480

% Box size and points

W=0.087;
H=0.275;
D=0.150;
dim = [W;D;H];

new_points = false;

try
    load('../data/points.mat')
    if(new_points)
        error('New data set will be created')
    end

catch
    i7_R = imread('../data/i7_R_axis.png');
    
    set(gcf, 'Position', get(0, 'Screensize'));
    
    
    image(i7_R)
    hold on
    title('Select points of image for calibration')
    hold off
    
    [X_R, Y_R] = getpts
    
    i7_L = imread('../data/i7_L_axis.png');
    
    image(i7_L)
    hold on
    title('Select points of image for calibration')
    hold off
    
    [X_L, Y_L] = getpts
    
    close all
    
    if(length(X_L) == 8 && length(Y_L) == 8 && length(X_R) == 8 && length(Y_R) == 8)    
        save('../data/points.mat', "X_R", "Y_R", "X_L", "Y_L")
    else
        fprintf('Not such points')
        return
    end
end
%% 
% Rounding values of $x$ and $y$ coordinates to get pixels

X_L_round = ceil(X_L)
X_R_round = ceil(X_R)
Y_L_round = ceil(Y_L)
Y_R_round = ceil(Y_R)
%% 
% Getting points in image coordinates

p_L = [X_L_round'; Y_L_round'; ones(1,8)]
p_R = [X_R_round'; Y_R_round'; ones(1,8)]
%% 
% Calculating rotation matrix, translation vector and intrinsic parameters matrix 
% uting |RTAa| function

[R_L_est,t_L_est,A_L_est] = RTAa(p_L,dim)
[R_R_est,t_R_est,A_R_est] = RTAa(p_R,dim)
%% 
% Getting the intrinsinc parameters matrix by the average of the two realizations:

A_est = (A_R_est+A_L_est)./2
% Homogeneous transforms
% Construction of homogeneous transform matrix for right view camera and ground 
% and for left view camera and groud.

T_R_0 = [R_R_est t_R_est; [0 0 0 1]]
T_L_0 = [R_L_est t_L_est; [0 0 0 1]]
%% 
% Given the transform between cameras using homogeneous transform algebra

T_R_L = T_R_0\T_L_0
T_L_R = T_L_0\T_R_0
%% 
% Returning to the rotation matrix and the translation vector
% 
% Right camera to left camera:

R_R_L = T_R_L([1:3],[1:3])
t_R_L = T_R_L([1:3],4)
%% 
% Left camera to right camera:

R_L_R = T_L_R([1:3],[1:3])
t_L_R = T_L_R([1:3],4)
