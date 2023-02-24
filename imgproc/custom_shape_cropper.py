#

from PIL import Image, ImageOps


# Convert to grayscale
def crop(source, mask):
    # mask = Image.open('mask.png').convert('L')

    # Threshold and invert the colors (white will be transparent)
    mask = mask.point(lambda x: x < 100 and 255)

    # The size of the images must match before apply the mask
    # img = ImageOps.fit(Image.open('source.png'), mask.size)
    img = ImageOps.fit(Image.open(source), mask.size)

    img.putalpha(mask)  # Modifies the original image without return

    # img.save('result.png')
    return img
