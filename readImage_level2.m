function imgOut = readImage(fn)

% fn = 'airport_12.jpg';
img = imread(fn);
img1 = imresize(img , [224,224]);
[R,C,~] = size(img1);
x = 'level2';    %'level1' or 'level2' or 'level3'

switch x
    case 'level1'
        imgOut = img1;
        
    case 'level2'
        r = 169;
        c = 169;
        
        x = floor((R - r)/2);
        y = floor((C - c)/2);
        w = r -1;
        h = c -1;
        imgOut = imcrop(img1,[x,y,w,h]);
        imgOut = rgb2gray(imgOut);
    case 'level3'
        r = 112;
        c = 112;
        
        x = floor((R - r)/2);
        y = floor((C - c)/2);
        w = r -1;
        h = c -1;
        imgOut = imcrop(img1,[x,y,w,h]);
        
end
end