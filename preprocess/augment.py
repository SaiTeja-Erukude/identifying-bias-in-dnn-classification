import os
from PIL import Image

def augment_img(img_path, result_folder_path):
    # augments the img and stores it in the result folder

    if not os.path.exists(img_path):
        return

    # Open the GIF file
    gif_image = Image.open(img_path)

    filename = img_path.split('/')[-1].split('.gif')[0]

    # Flip the image horizontally
    horizontal_flipped_image = gif_image.transpose(Image.FLIP_LEFT_RIGHT)
    horizontal_flipped_image.save(os.path.join(result_folder_path, filename + '_horizontal.JPEG'))

    # Flip the image vertically
    vertical_flipped_image = gif_image.transpose(Image.FLIP_TOP_BOTTOM)
    vertical_flipped_image.save(os.path.join(result_folder_path, filename + '_vertical.JPEG'))

    # save original gif img
    gif_image.save(os.path.join(result_folder_path, filename + '.JPEG'))
    
    print('Augmented: ', filename)


if __name__ == '__main__':

    input_path = '../../data/yale_faces/test_gif/'
    output_path = '../../data/yale_faces/test/'

    for filename in os.listdir(input_path):
        img_path = os.path.join(input_path, filename)
        augment_img(img_path, output_path)
