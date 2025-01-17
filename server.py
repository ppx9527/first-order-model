from flask import Flask, request, send_from_directory, make_response
from flask_cors import CORS
from crop_source import CropSource
from use_model import load_checkpoints, make_animation

import imageio
from skimage.transform import resize
from skimage import img_as_ubyte
import os
import filetype

app = Flask(__name__)
CORS(app, supports_credentials=True)


@app.route('/')
def index():
    return "Hello"


def allowed_file(filename, allowed):
    return '.' in filename and filename.rsplit('.', 1)[1] in allowed


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        try:
            token = request.args.getlist('token')[0]
        except IndexError:
            token = 0

        if file is not None and token is not None:
            if allowed_file(file.filename, ['png', 'jpg']):
                image = './temp/' + token + '-' + file.filename
                file.save(image)
                return '1'
            elif allowed_file(file.filename, ['mp4']):
                video = './temp/' + token + '-' + file.filename
                file.save(video)
                return '1'

            return '文件格式不对'
        else:
            return '服务器没有获取到文件'
    else:
        return '请求失败'


def generated(image, video, g):
    source_image = CropSource(image).crop_image()
    driving_video, fps = CropSource(video).crop_video()

    # 将图片的尺寸调整为256x256，并且将alpha通道移出
    source_image = resize(source_image, (256, 256))[..., :3]

    # 加载生成器和权重
    generator, kp_detector = load_checkpoints(
        config_path='config/vox-256.yaml',
        checkpoint_path='checkpoint/vox-cpk.pth.tar')

    # 前向计算
    predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=True)

    # 保存视频
    imageio.mimsave(g, [img_as_ubyte(frame) for frame in predictions], fps=fps)


@app.route('/getResult', methods=['GET', 'POST'])
def get_result():
    if request.method == 'GET':
        source = {}
        try:
            token = request.args.getlist('token')[0]
        except IndexError:
            token = 0

        for root, dirs, files in os.walk('./temp'):
            for file in files:
                if file.find(str(token), 0) == 0:
                    fullname = './temp/' + file
                    if filetype.guess(fullname).MIME.find('image') == 0:
                        source['image'] = fullname
                    elif filetype.guess(fullname).MIME.find('video') == 0:
                        source['video'] = fullname

        if len(source) != 2:
            return '没有获取到上传文件'

        source['generated'] = './temp/{}-g.mp4'.format(token)
        generated(source['image'], source['video'], source['generated'])

        directory = os.getcwd()
        response = make_response(send_from_directory(directory, source['generated'], as_attachment=True))
        response.headers["Content-Disposition"] = "attachment; filename={}".format(source['video'].encode().decode('latin-1'))

        for i in source.values():
            os.remove(i)

        return response

    else:
        return '请求失败'


if __name__ == '__main__':
    app.run()
