import face_recognition
import imageio
from skimage.transform import resize
import numpy


def extract_box(face_locations, shape, increase_area=0.3):
    """
    计算人脸裁剪区域
    :param face_locations: 人脸区域
    :param shape: 图片形状
    :param increase_area: 裁剪区域增加系数
    :return: 区域的位置
    """

    top, right, bottom, left = face_locations
    width = right - left
    height = bottom - top

    # 计算人脸区域截取增长的大小
    width_increase = max(increase_area, ((1 + 2 * increase_area) * height - width) / (2 * width))
    height_increase = max(increase_area, ((1 + 2 * increase_area) * width - height) / (2 * height))

    # 增加截取区域大小
    left = int(left - width_increase * width)
    top = int(top - height_increase * height)
    right = int(right + width_increase * width)
    bottom = int(bottom + height_increase * height)

    # 将截取窗口上移
    window = height_increase * height * 0.3
    top, bottom = int(top - window), int(bottom - window)

    top, bottom, left, right = max(0, top), min(bottom, shape[0]), max(0, left), min(right, shape[1])
    return top, right, bottom, left


class CropSource(object):
    def __init__(self, source):
        self.source = source

    def crop_image(self):
        """
        裁剪图片中的人脸
        :return: 包含人脸的图片
        """
        image = imageio.imread(self.source)
        image = image[..., :3]
        fl = face_recognition.face_locations(image, number_of_times_to_upsample=0, model='cnn')

        top, right, bottom, left = extract_box(fl[0], image.shape[:2])
        face_image = image[top:bottom, left:right]

        return face_image

    def crop_video(self):
        """
        裁剪视频
        :return: 裁剪好的视频
        """
        reader = imageio.get_reader(self.source)
        meta = reader.get_meta_data()
        fps = meta['fps']
        size = meta['source_size']
        frame_counts = meta['nframes']

        boxes = []
        frames = []
        valid_frames = []
        frame_count = 0

        try:
            for im in reader:
                frames.append(im)
                frame_count += 1

                if frame_count == frame_counts or len(frames) == 64:
                    batch_of_face_locations = face_recognition.batch_face_locations(frames, number_of_times_to_upsample=0)
                    
                    for frame_number_in_batch, face_locations in enumerate(batch_of_face_locations):
                        
                        if face_locations:
                            boxes.append(face_locations[0])
                            valid_frames.append(frames[frame_number_in_batch])
                    
                    frames = []
        except RuntimeError:
            pass

        boxes = numpy.mean(numpy.array(boxes), axis=0).astype(int)
        boxes = extract_box(boxes, size)

        if boxes:
            top, right, bottom, left = boxes

            face_video = []
            for f in valid_frames:
                face_video.append(resize(f[top:bottom, left:right], (256, 256)))

            return face_video, fps

if __name__ == '__main__':
    a = CropSource('temp/ggc.jpg').crop_image()
    print(a)
