import torch
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import random
import torchvision
from constant import *

class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        # self.operation_list = ["plus", "minus", "mult"]
        self.operation_list = ["plus", "minus"]
        self.term_num = 2
        self.upper_bound = 100
        self.num_per_op = (self.upper_bound ** self.term_num)

    def __getitem__(self, index):
        op = index // self.num_per_op
        index %= self.num_per_op
        a = index // self.upper_bound
        b = index % self.upper_bound
        if self.operation_list[op] == "plus":
            text_input = f"{a}+{b}="
            text_target = f"{a}+{b}={a+b}"
        elif self.operation_list[op] == "minus":
            text_input = f"{a}-{b}="
            text_target = f"{a}-{b}={a-b}"
        elif self.operation_list[op] == "mult":
            text_input = f"{a}*{b}="
            text_target = f"{a}*{b}={a*b}"
        else:
            assert False
        image_target, text_x, text_y = self.generage_image(text_target)
        image_input, _, _ = self.generage_image(text_input, text_x, text_y)
        image_input = torchvision.transforms.functional.to_tensor(image_input)
        image_target = torchvision.transforms.functional.to_tensor(image_target)
        return image_input, image_target

    def __len__(self):
        return len(self.operation_list) * self.num_per_op

    def generage_image(self, text, text_x=None, text_y=None):
        # 参考) https://qiita.com/implicit_none/items/a9bf7eebe125c9d773eb

        # 画像サイズ，背景色，フォントの色を設定
        canvas_size = IMAGE_SIZE[::-1]
        background_color = (255)
        text_color = (0)

        # 文字を描く画像の作成
        image = PIL.Image.new('L', canvas_size, background_color)
        draw = PIL.ImageDraw.Draw(image)

        fonts = [
            "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
            # "/usr/share/fonts/truetype/fonts-japanese-mincho.ttf",
            # "/usr/share/fonts/truetype/Gargi/Gargi.ttf",
            ]

        random_index = random.randrange(0, len(fonts))

        font = PIL.ImageFont.truetype(fonts[random_index], 18)

        # 用意した画像に文字列を描く
        text_width, text_height = draw.textsize(text, font=font)
        if text_x is None:
            # text_x = random.randrange(canvas_size[0] - text_width)
            text_x = 5
        if text_y is None:
            # text_y = random.randrange(canvas_size[1] - text_height)
            text_y = canvas_size[1] / 2 - text_height / 2
        text_pos = (text_x, text_y)  # 上下左右中央に配置
        assert text_pos[1] + text_height < IMAGE_HEIGHT
        assert text_pos[0] + text_width < IMAGE_WIDTH
        draw.text(text_pos, text, font=font, fill=text_color)

        return image, text_x, text_y
