import random
from torchvision import transforms
import torch
from PIL import Image
import numpy as np
def cut_paste(target_img_path, patch, area_ratio=[0.02, 0.15], aspect_ratio=0.3, color_jitter=0.2, transform=None):
    img = Image.open(target_img_path)
    patch = Image.open(patch);
    if color_jitter is None:
        jitter = None
    else:
        jitter = transforms.ColorJitter(brightness=color_jitter,
                                        contrast=color_jitter,
                                        saturation=color_jitter,
                                        hue=color_jitter)
    h = img.size[0]
    w = img.size[1]
    # ratio between area_ratio[0] and area_ratio[1]
    ratio_area = random.uniform(area_ratio[0], area_ratio[1]) * w * h
    # sample in log space
    log_ratio = torch.log(torch.tensor((aspect_ratio, 1 / aspect_ratio)))
    aspect = torch.exp(
        torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
    ).item()
    if color_jitter is not None:
        patch = jitter(patch)
    #补丁粘贴位置确认
    cut_w = patch.size[0]
    cut_h = patch.size[1]
    to_location_h = int(random.uniform(h/2 - cut_h, h/2 + cut_h))
    to_location_w = int(random.uniform(w/2 - cut_w, w/2 + cut_w))
    insert_box = [to_location_w, to_location_h, to_location_w + cut_w, to_location_h + cut_h]
    # img = np.array(img)
    # patch = np.array(patch)
    augmented_img = img.copy()
    #粘贴补丁
    augmented_img.paste(patch, insert_box)
    if transform is not None:
        augmented_img = transform(augmented_img)
    return augmented_img

from PIL import Image
import numpy as np

def blend_images(source_img_path, target_img_path, x_offset, y_offset, weight):
    # 打开源图像和目标图像
    source_img = Image.open(source_img_path)
    target_img = Image.open(target_img_path)

    # 将图像转换为NumPy数组以进行操作
    source_arr = np.array(source_img)
    target_arr = np.array(target_img)

    # 获取源图像的宽度和高度
    source_width, source_height = source_img.size

    # 将源图像局部融合到目标图像上
    for y in range(source_height):
        for x in range(source_width):
            target_x = x + x_offset
            target_y = y + y_offset

            # 检查目标像素是否在目标图像范围内
            if 0 <= target_x < target_img.width and 0 <= target_y < target_img.height:
                # 获取源图像像素的RGB值
                source_pixel = source_arr[y, x]
                target_pixel = target_arr[target_y, target_x]

                # 融合公式：new_pixel = (1 - weight) * target_pixel + weight * source_pixel
                new_pixel = tuple(int((1 - weight) * t + weight * s) for t, s in zip(target_pixel, source_pixel))

                # 将新像素赋值给目标图像数组
                target_arr[target_y, target_x] = new_pixel

    # 创建融合后的图像
    blended_img = Image.fromarray(target_arr)

    return blended_img

# if __name__ == "__main__":
#     source_img_path = "patch.jpg"  # 源图像的文件路径
#     target_img_path = "target.jpg"  # 目标图像的文件路径
#     x_offset = 400  # 源图像相对于目标图像的X偏移量
#     y_offset = 1250   # 源图像相对于目标图像的Y偏移量
#     weight = 0.5    # 融合权重，范围从0到1，0表示完全使用目标图像，1表示完全使用源图像
#
#     result_img = blend_images(source_img_path, target_img_path, x_offset, y_offset,weight)
#     result_img.save("result_image.jpg")  # 保存融合后的图像




if __name__ == '__main__':

    target_path = "./target.jpg"
    pathch_path = "./patch.jpg"
    img = cut_paste(target_img_path=target_path,patch=pathch_path)
    img.save("./images/aug_img.jpg")