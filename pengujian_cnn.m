clc; clear; close all; warning off all;

%%%% Busuk
% menetapkan nama folder citra
nama_folder = 'Data Uji\Busuk';
% membaca nama file yang berekstensi .jpg
nama_file = dir(fullfile(nama_folder,'*.jpg'));
% membaca jumlah file
jumlah_busuk = numel(nama_file);
% menginisialisasi variabel Img_busuk
Img_busuk = zeros(28,28,3,jumlah_busuk);
% melakukan pengolahan citra terhadap masing-masing file
for n = 1:jumlah_busuk
    % membaca citra rgb
    Img = im2double(imread(fullfile(nama_folder,nama_file(n).name)));
    % melakukan resizing citra
    Img_rsz = imresize(Img,[28 28]);
    % menyusun matriks Img_busuk
    Img_busuk(:,:,:,n) = Img_rsz;
end

%%%% Matang
% menetapkan nama folder citra
nama_folder = 'Data Uji\Matang';
% membaca nama file yang berekstensi .jpg
nama_file = dir(fullfile(nama_folder,'*.jpg'));
% membaca jumlah file
jumlah_matang = numel(nama_file);
% menginisialisasi variabel Img_matang
Img_matang = zeros(28,28,3,jumlah_matang);
% melakukan pengolahan citra terhadap masing-masing file
for n = 1:jumlah_matang
    % membaca citra rgb
    Img = im2double(imread(fullfile(nama_folder,nama_file(n).name)));
    % melakukan resizing citra
    Img_rsz = imresize(Img,[28 28]);
    % menyusun matriks Img_matang
    Img_matang(:,:,:,n) = Img_rsz;
end

%%%% Mengkal
% menetapkan nama folder citra
nama_folder = 'Data Uji\Mengkal';
% membaca nama file yang berekstensi .jpg
nama_file = dir(fullfile(nama_folder,'*.jpg'));
% membaca jumlah file
jumlah_mengkal = numel(nama_file);
% menginisialisasi variabel Img_mengkal
Img_mengkal = zeros(28,28,3,jumlah_mengkal);
% melakukan pengolahan citra terhadap masing-masing file
for n = 1:jumlah_mengkal
    % membaca citra rgb
    Img = im2double(imread(fullfile(nama_folder,nama_file(n).name)));
    % melakukan resizing citra
    Img_rsz = imresize(Img,[28 28]);
    % menyusun matriks Img_mengkal
    Img_mengkal(:,:,:,n) = Img_rsz;
end

%%%% Mentah
% menetapkan nama folder citra
nama_folder = 'Data Uji\Mentah';
% membaca nama file yang berekstensi .jpg
nama_file = dir(fullfile(nama_folder,'*.jpg'));
% membaca jumlah file
jumlah_mentah = numel(nama_file);
% menginisialisasi variabel Img_mentah
Img_mentah = zeros(28,28,3,jumlah_mentah);
% melakukan pengolahan citra terhadap masing-masing file
for n = 1:jumlah_mentah
    % membaca citra rgb
    Img = im2double(imread(fullfile(nama_folder,nama_file(n).name)));
    % melakukan resizing citra
    Img_rsz = imresize(Img,[28 28]);
    % menyusun matriks Img_mentah
    Img_mentah(:,:,:,n) = Img_rsz;
end

% menggabungkan citra pada masing2 kelas
Img_pisang = cat(4,Img_busuk,Img_matang,Img_mengkal,Img_mentah);

% men-setting nilai target (1=busuk, 2=matang, 3=mengkal, 4=mentah)
target_busuk = 1*ones(1,jumlah_busuk);
target_matang = 2*ones(1,jumlah_matang);
target_mengkal = 3*ones(1,jumlah_mengkal);
target_mentah = 4*ones(1,jumlah_mentah);
target_pisang = categorical([target_busuk,target_matang,target_mengkal,target_mentah]);

% proses pengujian CNN
load net
predictedLabels = classify(net,Img_pisang);

% menghitung nilai akurasi
test_acc = (sum(predictedLabels==target_pisang')/length(predictedLabels))*100;
disp(['Testing Accuracy : ',num2str(test_acc),' %'])
