import os
import random
import string

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


class GenerateVC(object):

    def __init__(self, width, height, save_path, font, font_size, distance):
        self.width = width
        self.height = height
        self.save_path = save_path
        self.font = font
        self.font_size = font_size
        self.distance = distance
        self.words = ''.join((string.ascii_letters, string.digits))
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    # 随机字符
    def random_word(self):
        return random.choice(self.words)

    # 背景颜色
    def bg_color(self):
        return random.randint(64, 200), random.randint(64, 200), random.randint(64, 200)

    # 生成验证码图片
    def generate_image(self, num):
        image = Image.new("RGB", (self.width, self.height), (255, 255, 255))
        # 创建Font对象
        font = ImageFont.truetype(self.font, self.font_size)
        # 创建Draw对象
        draw = ImageDraw.Draw(image)
        # 填充每一个像素
        for i in range(int((self.width * self.height) * 0.8)):
            xy = (random.randrange(0, self.width), random.randrange(0, self.height))
            draw.point(xy, fill=self.bg_color())

        # 输出文字
        size = self.width / 4
        word = ''
        for t in range(4):
            w = self.random_word()
            word = word + w
            draw.text((size * t + self.distance, 10), w, fill=(0, 0, 0), font=font)
        # 保存图片
        image.save(os.path.join(self.save_path, '%d_%s.jpg' % (num, word)), "jpeg")


if __name__ == '__main__':
    generateVC = GenerateVC(240, 60, "dataset/images", 'font/simfang.ttf', 38, 20)
    for i in tqdm(range(100000)):
        generateVC.generate_image(i)
