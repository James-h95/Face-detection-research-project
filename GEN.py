from PIL import Image
import random
import os

def generate_random_image(width, height):
    """
    生成一张随机的彩色图像，尺寸为 width x height。
    每个像素的 RGB 值都随机生成。
    """
    # 创建一个空的图像对象
    img = Image.new('RGB', (width, height))

    # 获取图像的像素对象
    pixels = img.load()

    # 填充每个像素点的颜色
    for i in range(width):
        for j in range(height):
            # 随机生成 RGB 值
            pixels[i, j] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    return img

def save_images(num_images, width, height, output_folder):
    """
    生成并保存 num_images 张随机彩色图像到指定文件夹。
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i in range(num_images):
        img = generate_random_image(width, height)
        img.save(os.path.join(output_folder, f"image_{i+1}.png"))
        print(f"Saved image_{i+1}.png")


if __name__ == "__main__":
    num_images = 350
    width = 64
    height = 128
    output_folder = "random_images"

    save_images(num_images, width, height, output_folder)
