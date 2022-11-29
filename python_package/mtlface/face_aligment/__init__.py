import numpy as np
from PIL import Image
import cv2
from skimage import transform as trans
from .mtcnn import detect_faces

def get_center_face(img):
    bounding_boxes, points = detect_faces(img)
    nrof_faces = len(bounding_boxes)
    if nrof_faces == 0:
        return
    # H, W
    img_size = np.asarray(img.size)
    bindex = 0
    if nrof_faces > 1:
        # x1, y1, x2, y2
        det = bounding_boxes
        bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
        img_center = img_size / 2
        offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1], (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
        offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
        bindex = np.argmax(bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
    # keeps only center faces.
    _bbox = bounding_boxes[bindex][:4]
    _landmark = points[bindex]
    return _bbox, _landmark


def face_process(img, output_size=112, plot=False):
    if isinstance(img, str):
        img = Image.open(img)
    if hasattr(img, 'filename'):
        fname = img.filename
    else:
        fname = None
    img = img.convert('RGB')
    results = get_center_face(img)
    if results is None:
        print('No face detected in {}.'.format(fname))
        return
    _landmark = results[1]
    aligned_img = face_alignment(np.array(img), _landmark, output_size, plot)
    aligned_img = Image.fromarray(aligned_img)
    return aligned_img


def face_alignment(img, landmarks, output_size, plot=False):
    # keypoints position, for 112*112
    src = np.array([[30.2946, 51.6963],
                    [65.5318, 51.5014],
                    [48.0252, 71.7366],
                    [33.5493, 92.3655],
                    [62.7299, 92.2041]], dtype=np.float32)
    src[:, 0] += 8
    base_size = 112

    if plot:
        for x, y in landmarks:
            img = cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0), -1)

    tform = trans.SimilarityTransform()
    tform.estimate(landmarks, src)
    M = tform.params[0:2, :]
    warped = cv2.warpAffine(img, M, (base_size, base_size), borderValue=0.0)
    return cv2.resize(warped, (output_size, output_size))
