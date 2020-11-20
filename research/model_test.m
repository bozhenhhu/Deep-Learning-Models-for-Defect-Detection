% Import the Layers
clc,clear;
netfile = 'huv27.h5'; 
classNames = {'0','1'}; 
% network = importKerasNetwork(netfile,'OutputLayerType','pixelclassification','ImageInputSize',[256,256]); 
network = importKerasNetwork(netfile); 
% figure
% plot(network);
% title(unet_pp)
data=importdata('defectnetinput.csv'); %test data's name is 20200812_007g.mgs
length=size(data);
sample_num=floor(length(2)/4)+10; 
test_img=data(:,sample_num);          %sample sample_num_th column
if length(1)==110592                  %384*288 or 640*480
    test_img=reshape(test_img,[384,288]);
elseif length(1)==307200
    test_img=reshape(test_img,[640,480]);
else
    disp('data length error');
end

%     data=data';
    test_img=imresize(test_img,[256,256]);    %args.height, args.width
    img=test_img/255.0;
    disp('data process successfully');

%Classify the image using the network 
y=predict(network, img); 
%Show the image and classification result 
figure(1);
imshow(test_img,[]);
figure(2);
imshow(y);
% title(['Classification result ' char(label)])