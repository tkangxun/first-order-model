import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.transform import resize
import warnings

from demo import load_checkpoints

from demo import make_animation
from skimage import img_as_ubyte



warnings.filterwarnings("ignore")

source_image = imageio.imread('/home/travis/PycharmProjects/Face-Off/results/example/target-face.png')
video_path = '/home/travis/PycharmProjects/first-order-model/checkpoints/dataset/08.mp4'

reader = imageio.get_reader(video_path)
fps = reader.get_meta_data()['fps']
driving_video = []
try:
    for im in reader:
        driving_video.append(im)
except RuntimeError:
    pass
reader.close()


# Resize image and video to 256x256

source_image = resize(source_image, (256, 256))[..., :3]
driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]



def display(source, driving, generated=None):
    fig = plt.figure(figsize=(8 + 4 * (generated is not None), 6))

    ims = []
    for i in range(len(driving)):
        cols = [source]
        cols.append(driving[i])
        if generated is not None:
            cols.append(generated[i])
        im = plt.imshow(np.concatenate(cols, axis=1), animated=True)
        plt.axis('off')
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=1000)
    plt.show(ani)

    return ani



display(source_image, driving_video)

generator, kp_detector = load_checkpoints(config_path='config/vox-256.yaml',
                            checkpoint_path='/home/travis/PycharmProjects/first-order-model/checkpoints/vox-cpk.pth.tar')

predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=True)
imageio.mimsave('./generated.mp4', [img_as_ubyte(frame) for frame in predictions], fps=fps)

display(source_image, driving_video, predictions)

print("shit")