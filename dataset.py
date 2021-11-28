import torch
import PIL.Image
import PIL.ImageDraw
import torchvision
from constant import *

class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.term_num = 2
        self.upper_bound = 10

    def __getitem__(self, index):
        a = index // self.upper_bound
        b = index % self.upper_bound
        text_input = f"{a}+{b}="
        text_target = f"{a}+{b}={a+b}"
        image_input = self.generage_image(text_input)
        image_target = self.generage_image(text_target)
        image_input = torchvision.transforms.functional.to_tensor(image_input)
        image_target = torchvision.transforms.functional.to_tensor(image_target)
        return image_input, image_target

    def __len__(self):
        return self.upper_bound ** self.term_num

    def generage_image(self, text):
        # 参考) https://qiita.com/implicit_none/items/a9bf7eebe125c9d773eb

        # 画像サイズ，背景色，フォントの色を設定
        canvasSize = IMAGE_SIZE
        background_color = (255)
        text_color = (0)

        # 文字を描く画像の作成
        image = PIL.Image.new('L', canvasSize, background_color)
        draw = PIL.ImageDraw.Draw(image)

        # 用意した画像に文字列を描く
        textWidth, textHeight = draw.textsize(text)
        textTopLeft = (canvasSize[0] // 6, canvasSize[1] // 2 - textHeight // 2)  # 前から1/6，上下中央に配置
        draw.text(textTopLeft, text, fill=text_color)

        return image
