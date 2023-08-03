clc;
clear;

path = '/Users/rhc/WorkSpace/Programs/RSStereo/Tmp/imgs/QC_left_570.tiff';
img_file = Tiff(path, 'r');
img_data = double(read(img_file));

min_data = min(min(img_data));
max_data = max(max(img_data));
img_data = (img_data - min_data) / (max_data - min_data);

imhist(img_data)
