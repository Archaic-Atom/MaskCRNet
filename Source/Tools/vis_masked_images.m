clc;
clear;

org_img_path = '/Users/rhc/WorkSpace/Programs/RSStereo/Tmp/imgs/3.png';
pred_img_path = '/Users/rhc/WorkSpace/Programs/RSStereo/Tmp/imgs/1.png';
mask_mat_path = '/Users/rhc/WorkSpace/Programs/RSStereo/Tmp/imgs/1.txt';
org_img=imread(org_img_path);
pred_img=imread(pred_img_path);
masks = load(mask_mat_path);



h = 448;
w = 448;
block_w = 16;
block_h = 16;

for i=1: length(masks)
    mask = masks(i);
    if mask < 0.5
        id = i-1;
        height_id = floor( id / floor(h / block_w));
        width_id = mod(id , floor(w / block_h));
        start_height = height_id * block_h + 1;
        end_height = height_id * block_h + block_h;
        start_width = width_id * block_w + 1;
        end_width = width_id * block_w + block_w;
        pred_img(start_height:end_height,...
            start_width: end_width,:) =  org_img(start_height:end_height,...
            start_width: end_width,:);
    else
        id = i-1;
        height_id = floor( id / floor(h / block_w));
        width_id = mod(id , floor(w / block_h));
        start_height = height_id * block_h + 1;
        end_height = height_id * block_h + block_h;
        start_width = width_id * block_w + 1;
        end_width = width_id * block_w + block_w;
        org_img(start_height:end_height,...
            start_width: end_width,:) =  127;
    end
end

% imshow(pred_img)
imwrite(pred_img,'/Users/rhc/WorkSpace/Programs/RSStereo/Tmp/imgs/4.png')
imwrite(org_img,'/Users/rhc/WorkSpace/Programs/RSStereo/Tmp/imgs/5.png')