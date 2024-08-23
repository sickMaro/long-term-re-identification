import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
sys.path.append('../methods/SOLIDER_REID')
from datasets.bases import read_image


def vis_detections(im, dets, thresh=0.5, show_text=True):
    """Draw detected bounding boxes."""
    class_name = 'face'
    inds = np.where(dets[:, -1] >= thresh)[0] if dets is not None else []
    if len(inds) == 0:
        return
    x0, y0, x1, y1 = dets[0][:4].astype(int)
    im2 = im[y0:y1, x0:x1]
    print(im2.shape)
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=2.5)
        )
        if show_text:
            ax.text(bbox[0], bbox[1] - 5,
                    '{:s} {:.3f}'.format(class_name, score),
                    bbox=dict(facecolor='blue', alpha=0.5),
                    fontsize=10, color='white')
    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                 fontsize=10)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('out.png')
    plt.show()


def bbox_vote(det):
    order = det[:, 4].ravel().argsort()[::-1]
    det = det[order, :]
    dets = None
    while det.shape[0] > 0:
        # IOU
        area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[0] + area[:] - inter)
        # get needed merge det and delete these det
        merge_index = np.where(o >= 0.3)[0]
        det_accu = det[merge_index, :]
        det = np.delete(det, merge_index, 0)
        if merge_index.shape[0] <= 1:
            continue
        det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
        max_score = np.max(det_accu[:, 4])
        det_accu_sum = np.zeros((1, 5))
        det_accu_sum[:, 0:4] = np.sum(det_accu[:, 0:4], axis=0) / np.sum(det_accu[:, -1:])
        det_accu_sum[:, 4] = max_score
        try:
            dets = np.row_stack((dets, det_accu_sum))
        except:
            dets = det_accu_sum
    if dets is not None:
        dets = dets[0:750, :]
    return dets


def add_borders(curr_img, target_shape=(224, 224), fill_type=0):
    curr_h, curr_w = curr_img.shape[0:2]
    shift_h = max(target_shape[0] - curr_h, 0)
    shift_w = max(target_shape[1] - curr_w, 0)

    image = cv2.copyMakeBorder(curr_img, shift_h // 2, (shift_h + 1) // 2, shift_w // 2, (shift_w + 1) // 2, fill_type)
    return image, shift_h, shift_w


def resize_image(images, target_size, interpolation=3):
    length = len(images)
    images_lst = np.zeros((length, target_size[0], target_size[1], 3))
    scale_lst = np.zeros((length, 4))

    for i, image in enumerate(images):
        curr_img_size = image.shape[0:2]

        images_lst[i] = (
            cv2.resize(image, (target_size[1], target_size[0]), interpolation=interpolation))

        resize_factor_y = target_size[1] / curr_img_size[1]
        resize_factor_x = target_size[0] / curr_img_size[0]

        scale_lst[i] = (target_size[1] / resize_factor_y,
                        target_size[0] / resize_factor_x,
                        target_size[1] / resize_factor_y,
                        target_size[0] / resize_factor_x)

    return images_lst, scale_lst


def extract_faces(img_size, detections, img_path, query_from_gui):
    faces = []
    detections_per_image = []
    for i in range(detections.shape[0]):
        if img_path[i] != '':
            image = read_image(img_path[i])
            print(image.size())
        else:
            if query_from_gui is None:
                raise RuntimeError('Query from gui is None')
            image = Image.fromarray(query_from_gui)

        image = image.resize((img_size[1], img_size[0]))

        for j in range(detections[i].shape[0]):
            x0, y0, x1, y1 = detections[i][j, :4].astype(int)
            # face = image[i, :, y0:y1, x0:x1]
            face = image.crop((x0, y0, x1, y1))
            # face = image[y0:y1, x0:x1]
            faces.append(face)
        detections_per_image.append(detections[i].shape[0])

    return faces, detections_per_image
