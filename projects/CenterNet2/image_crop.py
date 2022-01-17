import glob
import cv2
import os



def ratio_convert(cord, ratio):
    return cord[0]*ratio, cord[1]*ratio, cord[2]*ratio, cord[3]*ratio

def c_xywh2xyxy(x,y,w,h):
    return x-0.5*w, y-0.5*h, x+0.5*w, y+0.5*h

def xywh2xyxy(x,y,w,h):
    return x, y, x+w, y+h

def xyxy2c_xywh(x_min, y_min, x_max, y_max):
    return (x_min+x_max)*0.5, (y_min+y_max)*0.5, x_max-x_min, y_max-y_min

def cropimage_nxn(n):
    img_files = glob.glob('../data/phill/images/*.png')

    for img_file in img_files:
        file_num = os.path.splitext(img_file)[0].split('/')[-1]
        image = cv2.imread(img_file)

        with open('../data/phill/labels/' + file_num + '.txt', 'r') as f:
            labels = f.readlines()

        num = 0
        for i in range(n):
            for j in range(n):
                if 700 % n != 0:
                    print('warning pixel not arranged')
                    break
                crped_image = image[int(i * (700 / n)):int((i + 1) * (700 / n)),
                              int(j * (700 / n)):int((j + 1) * (700 / n))]
                cv2.imwrite('../data/phill/iimages/' + file_num + '_' + str(num) + '.png', crped_image)

                crped_labels = []
                for k in labels:
                    c, x, y, w, h = map(float, k.split(' '))
                    x_min, y_min, x_max, y_max = c_xywh2xyxy(x, y, w, h)

                    x_min_ = x_min if j * (1 / n) <= x_min and x_min <= (j + 1) * (1 / n) else False
                    x_max_ = x_max if j * (1 / n) <= x_max and x_max <= (j + 1) * (1 / n) else False
                    y_min_ = y_min if i * (1 / n) <= y_min and y_min <= (i + 1) * (1 / n) else False
                    y_max_ = y_max if i * (1 / n) <= y_max and y_max <= (i + 1) * (1 / n) else False
                    if not ((x_min_ or x_max_) and (y_min_ or y_max_)):
                        continue
                    else:
                        x_min = x_min if j * (1 / n) <= x_min and x_min <= (j + 1) * (1 / n) else j * (1 / n)
                        x_max = x_max if j * (1 / n) <= x_max and x_max <= (j + 1) * (1 / n) else (j + 1) * (1 / n)
                        y_min = y_min if i * (1 / n) <= y_min and y_min <= (i + 1) * (1 / n) else i * (1 / n)
                        y_max = y_max if i * (1 / n) <= y_max and y_max <= (i + 1) * (1 / n) else (i + 1) * (1 / n)

                        x_min = x_min - j * (1 / n)
                        x_max = x_max - j * (1 / n)
                        y_min = y_min - i * (1 / n)
                        y_max = y_max - i * (1 / n)

                        x, y, w, h = ratio_convert(*xyxy2c_xywh(x_min, y_min, x_max, y_max), ratio=n)
                        crped_labels.append(' '.join(map(str, [c, x, y, w, h])) + '\n')
                with open('../data/phill/llabels/' + file_num + '_' + str(num) + '.txt', 'w') as f:
                    f.writelines(crped_labels)

                num += 1


# compute coordinates by percentage
def crop_window_bbox(img, win_xyxy, window, labels):
    win_labels = []
    img_shape = img.shape[0]
    win_xyxy = [i/700 for i in win_xyxy]
    for label in labels:
        cls, *xywh = map(float, label.split(' '))
        xyxy = list(c_xywh2xyxy(*xywh))

        xyxy_= [0,0,0,0]
        xyxy_[0] = xyxy[0] if win_xyxy[0] <= xyxy[0] and xyxy[0] <= win_xyxy[2] else False
        xyxy_[1] = xyxy[1] if win_xyxy[1] <= xyxy[1] and xyxy[1] <= win_xyxy[3] else False
        xyxy_[2] = xyxy[2] if win_xyxy[0] <= xyxy[2] and xyxy[2] <= win_xyxy[2] else False
        xyxy_[3] = xyxy[3] if win_xyxy[1] <= xyxy[3] and xyxy[3] <= win_xyxy[3] else False

        if not ((xyxy_[0] or xyxy_[2]) and (xyxy_[1] or xyxy_[3])):
            continue

        else:
            xyxy[0] = xyxy[0] if win_xyxy[0] <= xyxy[0] and xyxy[0] <= win_xyxy[2] else win_xyxy[0]
            xyxy[1] = xyxy[1] if win_xyxy[1] <= xyxy[1] and xyxy[1] <= win_xyxy[3] else win_xyxy[1]
            xyxy[2] = xyxy[2] if win_xyxy[0] <= xyxy[2] and xyxy[2] <= win_xyxy[2] else win_xyxy[2]
            xyxy[3] = xyxy[3] if win_xyxy[1] <= xyxy[3] and xyxy[3] <= win_xyxy[3] else win_xyxy[3]

            xyxy[0] = xyxy[0] - win_xyxy[0]
            xyxy[1] = xyxy[1] - win_xyxy[1]
            xyxy[2] = xyxy[2] - win_xyxy[0]
            xyxy[3] = xyxy[3] - win_xyxy[1]

            xywh_n = ratio_convert(*xyxy2c_xywh(*xyxy), ratio=img_shape/window.shape[0])

            if xywh_n[2] <= 0.05 or xywh_n[3] <= 0.05:
                continue
            else:
                win_labels.append(' '.join(map(str, [cls, *xywh_n])) + '\n')

    return win_labels


def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0]-stepSize, stepSize):
        for x in range(0, image.shape[1]-stepSize, stepSize):
            # yield the current window
            yield (x, y, x+windowSize[0], y+windowSize[1],
                   image[y:y + windowSize[1], x:x + windowSize[0]])


def plot_box_txt(file_num):
    color = (0,0,0)
    for i in range(16):
        txt = '../data/phill/llabels/'+str(file_num)+'_'+str(i)+'.txt'
        img = '../data/phill/iimages/'+str(file_num)+'_'+str(i)+'.png'
        num = txt.split('/')[-1].replace('.txt','')
        img = cv2.imread(img)
        with open(txt,'r') as f:
            labels = f.readlines()
        for j in labels:
            j = list(map(float, j.split(' ')))
            x1,y1,x2,y2 = c_xywh2xyxy(j[1],j[2],j[3],j[4])
            c1, c2 = (int(x1*250), int(y1*250)), (int(x2*250), int(y2*250))
            cv2.rectangle(img, c1, c2, color, thickness=1)
        cv2.imwrite('../data/phill/'+str(file_num)+'_'+str(i)+'.png', img)


if __name__ == '__main__':
    img_files = glob.glob('../data/phill/images/*.png')

    for img_file in img_files:
        file_num = os.path.splitext(img_file)[0].split('/')[-1]
        image = cv2.imread(img_file)

        with open('../data/phill/labels/' + file_num + '.txt', 'r') as f:
            labels = f.readlines()

        win_iter = sliding_window(image, 150, (250, 250))
        for i, (*win_xyxy, win_img) in enumerate(win_iter):
            cv2.imwrite('../data/phill/iimages/' + file_num + '_' + str(i) + '.png', win_img)

            win_labels = crop_window_bbox(image, win_xyxy, win_img, labels)

            with open('../data/phill/llabels/' + file_num + '_' + str(i) + '.txt', 'w') as f:
                f.writelines(win_labels)

    # plot_box_txt(14000079)




