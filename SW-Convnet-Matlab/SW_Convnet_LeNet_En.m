%%% MATLAB Implementation of LeNet-5
%%% Author: xd.wp
%%% Date: 2016.10.22 14:29
%%% MATLAB Implementation of Scattering Wavelet + LeNet-5
%%% Author: Jun Tan
%%% Date: 2023.11.23 14:29
%% Program Description
% 1. Pooling: 2×2 average pooling is adopted
% 2. Network node description:
%                          Input layer: 28×28
%                          Layer 1: 24×24 (convolution) × 20
%                          Activation function: tanh
%                          Layer 2: 12×12 (pooling) × 20
%                          Layer 3: 100 (fully connected)
%                          Layer 4: 10 (softmax)
% 3. Network training: 800 samples are used; Model validation: 100 samples are used
% 4. Added scattering wavelet filtering. Feature extraction is implemented using sf.featureMatrix(XS)
clear all;clc;
%% Network initial 
layer_c1_num=20;
layer_s1_num=20;
layer_f1_num=100;
layer_output_num=10;
% Weight adjustment step size
yita = 0.01;
% Bias initialization
bias_c1 = (2 * rand(1, 20) - ones(1, 20)) / sqrt(20);
bias_f1 = (2 * rand(1, 100) - ones(1, 100)) / sqrt(20);
% Convolution kernel initialization
[kernel_c1, kernel_f1] = init_kernel(layer_c1_num, layer_f1_num);
% Pooling kernel initialization
pooling_a = ones(2, 2) / 4;
% Weights of the fully connected layer
weight_f1=(2*rand(20,100)-ones(20,100))/sqrt(20);
weight_output=(2*rand(100,10)-ones(100,10))/sqrt(100);
% Scattering wavelet initialization
sf = waveletScattering(SignalLength=784);
disp('Network initialization completed......');
%% Start network training
disp('Start network training......');
for iter=1:20
for n=1:90
    for m=0:9
        %read data samples
        train_data=imread(strcat('img\',num2str(m),'_',num2str(n),'.bmp'));
        train_data=double(train_data);
        OLDdata=train_data;
        %wavelet layer
        XS=reshape(train_data,784,1);
        
        train_data = sf.featureMatrix(XS);
        train_data=train_data';
        NewTrain=zeros(28);
        N1=train_data(:,1:64);
        N1=reshape(N1,28,16);
        NewTrain(:,1:16)=N1;
        NewTrain(1:7,17:22)=train_data(:,65:70);
        train_data=OLDdata;
        % Mean subtraction
        % train_data = wipe_off_average(train_data);
        % Forward propagation: Enter Convolution Layer 1
        for k = 1:layer_c1_num
            if k <= layer_c1_num/2
                state_c1(:,:,k) = convolution(train_data, kernel_c1(:,:,k));
            else
                state_c1(:,:,k) = convolution(NewTrain, kernel_c1(:,:,k));
            end
            %state_c1(:,:,k) = convolution(train_data, kernel_c1(:,:,k));
            % Enter activation function
            state_c1(:,:,k) = tanh(state_c1(:,:,k) + bias_c1(1,k));
            % Enter Pooling Layer 1
            state_s1(:,:,k) = pooling(state_c1(:,:,k), pooling_a);
        end
        % Enter Fully Connected Layer F1
        [state_f1_pre, state_f1_temp] = convolution_f1(state_s1, kernel_f1, weight_f1);
        % Enter activation function
        for nn=1:layer_f1_num
            state_f1(1,nn)=tanh(state_f1_pre(:,:,nn)+bias_f1(1,nn));
        end
        % Enter Softmax layer
        for nn = 1:layer_output_num
            output(1, nn) = exp(state_f1 * weight_output(:, nn)) / sum(exp(state_f1 * weight_output));
        end
        %% Error calculation section
        Error_cost = -output(1, m + 1);
        % if (Error_cost < -0.98)
        %     break;
        % end
        %% Parameter adjustment section
        [kernel_c1,kernel_f1,weight_f1,weight_output,bias_c1,bias_f1]=CNN_upweight(yita,Error_cost,m,train_data,...
                                                                                                state_c1,state_s1,...
                                                                                                state_f1,state_f1_temp,...
                                                                                                output,...
                                                                                                kernel_c1,kernel_f1,weight_f1,weight_output,bias_c1,bias_f1);
    end    
end
iter
end
disp('Network train completed, start model validation ...');
count=0;
for n=90:100
    for m=0:9
        %read test data sample
        train_data=imread(strcat('img\',num2str(m),'_',num2str(n),'.bmp'));
        train_data=double(train_data);
        OLDdata=train_data;
         %wavelet layer
        XS=reshape(train_data,784,1);
        %sf = waveletScattering(SignalLength=784);
        train_data = sf.featureMatrix(XS);
        train_data=train_data';
        NewTrain=zeros(28);
        N1=train_data(:,1:64);
        N1=reshape(N1,28,16);
        NewTrain(:,1:16)=N1;
        NewTrain(1:7,17:22)=train_data(:,65:70);
        train_data=OLDdata;
        % Forward propagation: Enter Convolution Layer 1
        for k=1:layer_c1_num
            if k<=layer_c1_num/2
                state_c1(:,:,k)=convolution(train_data,kernel_c1(:,:,k));
            else
                state_c1(:,:,k)=convolution(NewTrain,kernel_c1(:,:,k));
            end
            % state_c1(:,:,k)=convolution(train_data,kernel_c1(:,:,k));
            % Enter activation function
            state_c1(:,:,k)=tanh(state_c1(:,:,k)+bias_c1(1,k));
            % Enter Pooling Layer 1
            state_s1(:,:,k)=pooling(state_c1(:,:,k),pooling_a);
        end
        % Enter Fully Connected Layer F1
        [state_f1_pre,state_f1_temp]=convolution_f1(state_s1,kernel_f1,weight_f1);
        % Enter activation function
        for nn=1:layer_f1_num
            state_f1(1,nn)=tanh(state_f1_pre(:,:,nn)+bias_f1(1,nn));
        end
        % Enter softmax layer
        for nn=1:layer_output_num
            output(1,nn)=exp(state_f1*weight_output(:,nn))/sum(exp(state_f1*weight_output));
        end
        [p,classify]=max(output);
        if (classify==m+1)
            count=count+1;
        end
        fprintf('True label: %d  Network prediction: %d  Probability value: %f \n', m, classify - 1, p);
    end
end