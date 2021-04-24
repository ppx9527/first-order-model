import face_recognition
import imageio
from skimage.transform import resize


class CropSource(object):
    def __init__(self, source, cuda=False):
        self.source = source
        self.cuda = cuda

    def crop_image(self):
        image = imageio.imread(self.source)
        top, right, bottom, left = self.extract_box(image)
        face_image = image[top:bottom, left:right]
        return face_image

    def crop_video(self):
        reader = imageio.get_reader(self.source)
        fps = reader.get_meta_data()['fps']

        frames = []

        try:
            for im in reader:
                frames.append(im)
        except RuntimeError:
            pass

        face_video = []

        for f in frames:
            top, right, bottom, left = self.extract_box(f)
            face_video.append(resize(f[top:bottom, left:right], (256, 256)))

        return face_video, fps

    def extract_box(self, frame, increase_area=0.15):
        """
        获取人脸裁剪区域
        :param frame: 图片
        :param increase_area: 裁剪区域增加系数
        :return: 区域的位置
        """
        frame = frame[..., :3]

        if self.cuda:
            face_locations = face_recognition.face_locations(frame, number_of_times_to_upsample=0, model='cnn')
        else:
            face_locations = face_recognition.face_locations(frame)

        if face_locations:
            top, right, bottom, left = face_locations[0]
            width = right - left
            height = bottom - top

            # 计算人脸区域截取增长的大小
            width_increase = max(increase_area, ((1 + 2 * increase_area) * height - width) / (2 * width))
            height_increase = max(increase_area, ((1 + 2 * increase_area) * width - height) / (2 * height))

            # 增加截取区域大小，使其包括头发
            left = int(left - width_increase * width)
            top = int(top - height_increase * height * 2)
            right = int(right + width_increase * width)
            bottom = int(bottom + height_increase * height)

            top, bottom, left, right = max(0, top), min(bottom, frame.shape[0]), max(0, left), min(right, frame.shape[1])

            return top, right, bottom, left
        else:
            return 0


if __name__ == '__main__':
    a = CropSource('temp/got-05.png').crop_image()
    a.crop_video()
