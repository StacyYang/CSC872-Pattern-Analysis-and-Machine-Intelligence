%INPUT:
%     I  = input image(gray or rgb, process time will vary due to the
%          difference of image sizes.
%     h  = bandwidth
%     TH = stop threshold
%     K  = max iteration
%---------------------------------------------------------------------
clear 
close all
clc

%%%%LOADING IMAGES
Directory = './'; 
A = dir(fullfile(Directory, 'Baboon.bmp'));
File = fullfile(Directory, A.name);
ImageTest = imread(File);
ImageTest = imresize(ImageTest, [64,64]);
imgTest = double(ImageTest);

figure('Name', 'Original image');
imshow(imgTest, [])
result = mean_shift_filter(imgTest, 0.5, 15 , 10);
figure('Name', 'Result');
imshow(result, []);

%%%%GAUSSIAN(ISOTROPIC) KERNEL
function weight = gaussian_kernel(x, xn, h)
     weight = exp(- norm((x(:) - xn(:)) / h));
end

%%%%MEAN SHIFT PROCEDURE
function Output = mean_shift_filter(I, TH, K, h)
    % pixel value of dimension 1 or 3
    dims = size(I);
    Output = zeros(dims);
    for y = 1:dims(2)
        for x = 1:dims(1)
            yk = I(y, x, :);                         %current pixel
            shift = TH + 1;
            k = 1;
            while sum(shift) > TH && k <= K
                numerator = 0;
                denominator = 0;
                for j = 1:dims(2)                    %loop through all the pixels 
                    for i = 1:dims(1)
                        weight = gaussian_kernel(yk, I(j,i,:), h);
                        if weight < h
                            numerator = numerator + I(j, i) * weight;
                            denominator = denominator + weight;
                        end
                    
                    end
                end
                yk1 = numerator ./ denominator;
                shift = norm(yk1(:) - yk(:));
                yk = yk1;
                k = k + 1;
            end
            Output(y, x, :) = yk;
        end
    end
end



