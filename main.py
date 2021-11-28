import PIL.Image
import PIL.ImageDraw


def generage_image(text):
    # 参考) https://qiita.com/implicit_none/items/a9bf7eebe125c9d773eb

    # 画像サイズ，背景色，フォントの色を設定
    canvasSize = (300, 150)
    backgroundRGB = (255, 255, 255)
    textRGB = (0, 0, 0)

    # 文字を描く画像の作成
    image = PIL.Image.new('RGB', canvasSize, backgroundRGB)
    draw = PIL.ImageDraw.Draw(image)

    # 用意した画像に文字列を描く
    textWidth, textHeight = draw.textsize(text)
    textTopLeft = (canvasSize[0] // 6, canvasSize[1] // 2 - textHeight // 2)  # 前から1/6，上下中央に配置
    draw.text(textTopLeft, text, fill=textRGB)

    return image


texts = ["1+1=2", "2+2=4", "3+3=6"]

for i, text in enumerate(texts):
    image = generage_image(text)
    image.save(f"{i}.png")
