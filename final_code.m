%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                            Br41n Hackathon 2022                         %
%                              ECoG Hand Pose                             %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all
clear all
clc

%% LOading Dataset
load('ECoG_Handpose.mat')

%cut the beginning samples through visual inspection (first 14497 samples)
y = y(:,14497:end);

time = y(1,:);
classes = y(62,:);
signal = y(2:61,:);
glove_signal = y(63:end,:);

fs = 1200;               %sampling frequency
fNy=fs/2;                %Nyquist frequency
T= length(signal)/fs;    %Period of the signal
dt = time(2)-time(1);    %sampling interval time

%look for the label position
pos0 = find(classes==0);
pos1 = find(classes==1);
pos2 = find(classes==2);
pos3 = find(classes==3);

%% ------------------------REFERENCING---------------------------
%Common average reference (CAR) spatial filter to remove the
%global influence (change in heart rate, respiratory
%influences).

%calcolo il filtro
CAR = mean(signal);

for ii=1:length(signal)
    signal_CAR(:,ii) = signal(:,ii) - CAR(1,ii);
end

%Decide wheter to use CAR or not:
%car_active = 1 --> Use car filter
%car_active = 0 --> do not use car filter

car_active = 1; 
if (car_active==0)
    signal_CAR = signal;
end
    
%% Filter ECoG Signal
for j=1:length(signal(:,1))
    
    xx = signal_CAR(j,:);
     
    %Notch cascade filter Butterworth order 5
    harmonic = 6;
    for jj=1:harmonic       
    ordine = 5;
    ft = 50*jj;
    [b,a]=butter(ordine,[(ft-5)/fNy (ft+5)/fNy],'stop');  %se voglio passa alto
    %freqz(b,a,500,fs)
    xx = filtfilt(b,a,xx);
    end
    %plot(xx)
    
    %%Passband Filter between 100 Hz and 500 Hz
    ordine=40;
    ft=1200;
    fNy=ft/2;
    [bb,aa] = butter(ordine/2,[100/fNy 500/fNy],'bandpass');
    %freqz(bb,aa,ft,ft)
    signal_100hz_CAR(j,:) = filtfilt(bb,aa,xx);
    %plot(signal_100hz(j,:))
    
    %%Passband filter between 50 and 300Hz (remove DC offset)
    ordine=10;
    ft=1200;
    fNy=ft/2;
    [bbb,aaa] = butter(ordine/2,[50/fNy 300/fNy],'bandpass');
    %freqz(bbb,aaa,ft,ft)
    signal_1hz_CAR(j,:) = filtfilt(bbb,aaa,xx);    
end

%add the classes
signal_100hz_CAR(61,:) = classes;
signal_1hz_CAR(61,:) = classes;

%load('signal_1hz_CAR.mat')

y = signal_1hz_CAR(1:60,:);
classes = signal_1hz_CAR(61,:);
cue = find(diff(classes)~=0);
cue(1,180) = 492529;
fs = 1200;
fNy=fs/2;
time_step = fs/16;

%channel_selection
canali = [1:60];

y = y(canali,:);

%% Trial Division
j=1;
trial_labels(1,j) = classes(cue(j));
trial_matrix(:,:,j) = y(:,1:cue(j));

for j=2:180
    trial_labels(1,j) = classes(cue(j));
    
    if trial_labels(1,j)==0
    trial_matrix(:,:,j) = y(:,cue(j)-2399:cue(j));
    else
    trial_matrix(:,:,j) = y(:,cue(j-1):cue(j-1)+2399);
    end
end


%% feature extraction - 3 bandpass filter
 ordine = 4;
 
%banda 60-90 Hz
[b1,a1] = butter(ordine,[60/fNy 90/fNy],'bandpass');
for i=1:length(canali)
    y1(i,:) = filtfilt(b1,a1,y(i,:));
end

%banda 110-140 Hz
[b2,a2] = butter(ordine,[110/fNy 140/fNy],'bandpass');
for i=1:length(canali)
    y2(i,:) = filtfilt(b2,a2,y(i,:));
end

%banda 160-190 Hz
[b3,a3] = butter(ordine,[160/fNy 190/fNy],'bandpass');
for i=1:length(canali)
    y3(i,:) = filtfilt(b3,a3,y(i,:));
end

%% Trial division - through the bandwidth
j=1;
trial_labels(1,j) = classes(cue(j));
trial_matrix1(:,:,j) = y1(:,1:cue(j));
trial_matrix2(:,:,j) = y2(:,1:cue(j));
trial_matrix3(:,:,j) = y3(:,1:cue(j));

for j=2:180
    trial_labels(1,j) = classes(cue(j));
    
    %first bandwidth
    if trial_labels(1,j)==0
    trial_matrix1(:,:,j) = y1(:,cue(j)-2399:cue(j));
    else
    trial_matrix1(:,:,j) = y1(:,cue(j-1):cue(j-1)+2399);
    end
    
    %second bandwidth
    if trial_labels(1,j)==0
    trial_matrix2(:,:,j) = y2(:,cue(j)-2399:cue(j));
    else
    trial_matrix2(:,:,j) = y2(:,cue(j-1):cue(j-1)+2399);
    end
    
    %third bandwidth
    if trial_labels(1,j)==0
    trial_matrix3(:,:,j) = y3(:,cue(j)-2399:cue(j));
    else
    trial_matrix3(:,:,j) = y3(:,cue(j-1):cue(j-1)+2399);
    end
end

%% FEATURE EXTRACTION from windows 
finestra = [1:2400];

for k=1:180
    for ii=1:length(canali)
        %power estimation (squaring)
        pow_estim = trial_matrix1(ii,finestra,k).^2;
        %temporal average 500 ms
        feature1(ii) = log(mean(pow_estim));
    end
    for ii=1:length(canali)
        %power estimation (squaring)
        pow_estim = trial_matrix2(ii,finestra,k).^2;
        %temporal average 500 ms
        feature2(ii) = log(mean(pow_estim));
    end
    for ii=1:length(canali)
        %power estimation (squaring)
        pow_estim = trial_matrix3(ii,finestra,k).^2;
        %temporal average 500 ms
        feature3(ii) = log(mean(pow_estim));
    end
feature_vector(k,:) = [feature1 feature2 feature3];
end

%% classification
%% classification
bag_ada='bag';
numTrees=50;
numFolds=5;

ACC_test=cell(numFolds+1,5+2);
ACC_train=cell(numFolds+1,5+2);
    
pos_move = find(trial_labels~=0);
feature_vector = feature_vector([pos_move],:);
trial_labels = trial_labels(:,[pos_move]);

posizioni = crossvalind('Kfold',90,5);

for i=1:numFolds
    k=i;
    training = feature_vector([find(posizioni~=i)],:);
    train_labels = trial_labels(:,[find(posizioni~=i)]);

    testing = feature_vector([find(posizioni==i)],:);
    test_labels = trial_labels(:,[find(posizioni==i)]);

%    predicted_labels = classify2(testing,training,train_labels','diaglinear')';
    [ I,MDL,Y,Score,Accuracy,I_train,Y_train,Score_train,Acc_train ] = runRFnew( train_labels', training, test_labels', testing,numTrees, bag_ada  );
    stats = confusionmatStats(test_labels',Y);
        ACC_test{k,1}=I;
        ACC_test{k,2}=MDL;
        ACC_test{k,3}=test_labels';
        
        ACC_test{k,4}=Y;
        ACC_test{k,5}=Score;
        ACC_test{k,6}=stats.avgAcc;
        
        ACC_test{k,7}=stats.avgPrecision;
        ACC_test{k,8}=stats.avgRecall;
        ACC_test{k,9}=stats.fscore;
        ACC_test{k,10}=stats.weighted_fscore;
        

        stats = confusionmatStats(train_labels',Y_train);

        ACC_train{k,1}=I_train;
        ACC_train{k,2}=MDL;
        ACC_train{k,3}=train_labels';
        ACC_train{k,4}=Y_train;
        ACC_train{k,5}=Score_train;
        
        ACC_train{k,6}=stats.avgAcc;
        ACC_train{k,7}=stats.avgPrecision;
        ACC_train{k,8}=stats.avgRecall;
        ACC_train{k,9}=stats.fscore;
        ACC_train{k,10}=stats.weighted_fscore;
        
%     accuracy(i) = sum(predicted_labels == test_labels) / length(test_labels);
end

 ACC_test2 = meanStats(ACC_test,k);
 ACC_train2 = meanStats(ACC_train,k);
% accuracy_fold = mean(accuracy)
% std_accuracy = std(accuracy)

% mn=cell2mat(ACC_test(:,6));
% %     whos mn
%     ACC_test{k+1,7}=mean(mn);
%     ACC_test{k+1,8}=std(mn);
%     
%     mn2=cell2mat(ACC_train(:,6));
% %     whos mn
%     ACC_train{k+1,7}=mean(mn2);
%     ACC_train{k+1,8}=std(mn2);
    