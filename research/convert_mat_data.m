clear all
close all;
clc




Getfiles()

function[names] = Getfiles()
    filefolder = fullfile('./new_mat');
    diroutput = dir(fullfile(filefolder,'*.mat'));
    names = {diroutput.name};
    [~,num] = size(names);
    for i = 1:num
        mat_name = names(:,i);
        data1 = load(char(mat_name));
        disp(mat_name)
        data = data1.data;
        [m,~]=size(data);
        data = reshape(data,m,640,480); %388 284
        data = permute(data,[2,3,1]);
%         figure(10)
%         imagesc(data(:,:,50));
        save(char(mat_name),'data');
    end
end



