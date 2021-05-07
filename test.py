import face_recognition
import imageio
import numpy
from skimage.transform import resize
from skimage import img_as_ubyte

a = imageio.get_reader('temp/aaaa.mp4')
fps = a.get_meta_data()['nframes']
print(fps % 64)
fs = []
v = []
b = []

c = 0

for f in a:
    fs.append(f)
    c += 1
    if c == fps:
        print(len(fs))
    
    if len(fs) == 64:
        # batch_of_face_locations = face_recognition.batch_face_locations(fs, number_of_times_to_upsample=0)
        # print(len(batch_of_face_locations))
        # for frame_number_in_batch, face_locations in enumerate(batch_of_face_locations):
            
        #     if face_locations:
        #         b.append(face_locations[0])
        #         v.append(fs[frame_number_in_batch])
        
        fs = []

# b = numpy.mean(numpy.array(b), axis=0).astype(int)
# top, right, bottom, left = b

# ff = []
# for f in v:
#     ff.append(resize(f[top:bottom, left:right], (256, 256)))

# imageio.mimsave('r.mp4', [img_as_ubyte(frame) for frame in ff], fps=fps)