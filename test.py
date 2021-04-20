from operator import mod
import imageio
from skimage.transform import resize
from skimage import img_as_ubyte
from demo import load_checkpoints, make_animation

source_image = imageio.imread('asset/got-05.png')
reader = imageio.get_reader('asset/04.mp4')

#Resize image and video to 256x256

source_image = resize(source_image, (256, 256))[..., :3]

fps = reader.get_meta_data()['fps']
driving_video = []
try:
    for im in reader:
        driving_video.append(im)
except RuntimeError:
    pass
reader.close()

driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]


generator, kp_detector = load_checkpoints(
    config_path='config/vox-256.yaml',
    checkpoint_path='checkpoint/vox-cpk.pth.tar')

predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=True)

imageio.mimsave('./asset/r.mp4', [img_as_ubyte(frame) for frame in predictions], fps=fps)