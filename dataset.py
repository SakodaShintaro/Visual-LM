import torch
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import torchvision
from constant import *

class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.operation_list = ["plus", "minus", "mult"]
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
        image_input = self.generage_image(text_input)
        image_target = self.generage_image(text_target)
        image_input = torchvision.transforms.functional.to_tensor(image_input)
        image_target = torchvision.transforms.functional.to_tensor(image_target)
        return image_input, image_target

    def __len__(self):
        return len(self.operation_list) * self.num_per_op

    def generage_image(self, text):
        # 参考) https://qiita.com/implicit_none/items/a9bf7eebe125c9d773eb

        # 画像サイズ，背景色，フォントの色を設定
        canvasSize = IMAGE_SIZE
        background_color = (255)
        text_color = (0)

        # 文字を描く画像の作成
        image = PIL.Image.new('L', canvasSize, background_color)
        draw = PIL.ImageDraw.Draw(image)

        font = PIL.ImageFont.truetype("/usr/share/fonts/truetype/fonts-japanese-gothic.ttf", 16)

        # 用意した画像に文字列を描く
        textWidth, textHeight = draw.textsize(text, font=font)
        textTopLeft = (canvasSize[0] // 8, canvasSize[1] // 2 - textHeight // 2)  # 前から1/6，上下中央に配置
        assert textTopLeft[1] + textHeight < IMAGE_WIDTH
        assert textTopLeft[0] + textWidth < IMAGE_WIDTH
        draw.text(textTopLeft, text, font=font, fill=text_color)

        return image
