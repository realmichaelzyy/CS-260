function a = scale_save_img(img, filename)
    mx = max(img(:));
    mn = min(img(:));
    imgScaled = (img-mn)/(mx-mn);
    %# convert to uint8 and save
    imwrite( uint8(round(imgScaled*255)), filename);
end
