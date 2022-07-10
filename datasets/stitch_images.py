import json
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import PIL.Image as Image
import pickle

COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
              [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]


def plot_union_bbox(proc_anno, img, yn):

    union_bbox = proc_anno['hoi_union_bbox']
    subs = [proc_anno['bbox'][j] for j in proc_anno['hoi_sub']]
    objs = [proc_anno['bbox'][j] for j in proc_anno['hoi_obj']]

    # pts1 = np.float32([[xmin, ymin], [xmax, ymin], [xmin, ymax], [xmax, ymax]])
    # pts2 = np.float32([[0, 0], [xmax - xmin, 0], [0, ymax - ymin], [xmax - xmin, ymax - ymin]])
    # matrix = cv2.getPerspectiveTransform(pts1, pts2)
    # image_output = cv2.warpPerspective(image, matrix, (xmax - xmin, ymax - ymin))

    colors = COLORS * 100

    plt.figure()
    plt.imshow(img)

    ax = plt.gca()

    for (sub_xmin, sub_ymin, sub_xmax, sub_ymax), (obj_xmin, obj_ymin, obj_xmax, obj_ymax), c in zip(subs, objs, colors):

        ax.add_patch(plt.Rectangle((sub_xmin, sub_ymin), sub_xmax - sub_xmin, sub_ymax - sub_ymin,
                                   fill=False, color=c, linewidth=1))

        ax.add_patch(plt.Rectangle((obj_xmin, obj_ymin), obj_xmax - obj_xmin, obj_ymax - obj_ymin,
                                   fill=False, color=c, linewidth=1))

    if yn == 1:
        ax.add_patch(plt.Rectangle((union_bbox[0], union_bbox[1]), union_bbox[2] - union_bbox[0], union_bbox[3] - union_bbox[1],
                                   fill=False, color='r', linewidth=3))
    plt.axis('off')

    plt.show()


def get_random_index(sample_num, img_num, nohoi_index):

    index = np.arange(len(img_num))

    for i in range(len(nohoi_index)):
        if nohoi_index[i] in index:
            index.remove(nohoi_index[i])

    random_index = np.random.choice(index, sample_num, replace=True)

    return list(random_index)


def get_sim_index(sample_num, nohoi_index, sim_index):

    for i in range(len(nohoi_index)):
        if nohoi_index[i] in sim_index:
            sim_index.remove(nohoi_index[i])

    random_index = np.random.choice(sim_index, sample_num, replace=True)

    return list(random_index)


def get_basic_union_bbox(random_index, annotations, dataset):

    anno_bbox = []  # 每一行都是一张图片中所有目标的gt框
    anno_id = []    # 每一行都是上行中gt框对应的物体类别

    hoi_sub = []  # 每一行代表一张图片中所有的人在anno_box中的索引
    hoi_obj = []  # 每一行代表与上一个对应的人交互的物体在anno_box中的索引
    verb_id = []  # 这个动作的类别
    hoi_id = []   # 这个hoi的类别 verb+obj

    img_name = []

    hoi_union_bbox = []  # 每一行代表一张图片中所有HOI的union_bbox

    for i in range(len(random_index)):
        annotation = annotations[random_index[i]]['annotations']
        bbox = []
        category_id = []
        for anno in annotation:
            bbox.append(anno['bbox'])
            category_id.append(anno['category_id'])
        anno_bbox.append(bbox)
        anno_id.append(category_id)

        hoi_annotation = annotations[random_index[i]]['hoi_annotation']
        subject_id, object_id, category_id, hoi_category_id = [], [], [], []
        for hoi_anno in hoi_annotation:
            if hoi_anno['object_id'] != -1:
                subject_id.append(hoi_anno['subject_id'])
                object_id.append(hoi_anno['object_id'])
                category_id.append(hoi_anno['category_id'])
                if dataset=='hico':
                    hoi_category_id.append(hoi_anno['hoi_category_id'])
        hoi_sub.append(subject_id)
        hoi_obj.append(object_id)
        verb_id.append(category_id)
        hoi_id.append(hoi_category_id)

        union_bbox = []
        for sub_id, obj_id in zip(hoi_sub[i], hoi_obj[i]):
            try:
                xmin = min(anno_bbox[i][sub_id][0], anno_bbox[i][obj_id][0])
                ymin = min(anno_bbox[i][sub_id][1], anno_bbox[i][obj_id][1])
                xmax = max(anno_bbox[i][sub_id][2], anno_bbox[i][obj_id][2])
                ymax = max(anno_bbox[i][sub_id][3], anno_bbox[i][obj_id][3])
            except:
                raise Exception('当前报错index是：{},所有index是：{},当前bbox长度是:{},sub_id是:{},obj_id是:{}'.format((i, random_index[i]), random_index, len(anno_bbox[i]), sub_id, obj_id))

            union_bbox.append([xmin, ymin, xmax, ymax])

        hoi_union_bbox.append(union_bbox)

        img_name.append(annotations[random_index[i]]['file_name'])

    anno_dict = {}
    anno_dict['file_name'] = img_name
    anno_dict['bbox'] = anno_bbox
    anno_dict['bbox_id'] = anno_id
    anno_dict['hoi_sub'] = hoi_sub
    anno_dict['hoi_obj'] = hoi_obj
    anno_dict['verb_id'] = verb_id
    anno_dict['hoi_id'] = hoi_id
    anno_dict['hoi_union_bbox'] = hoi_union_bbox

    return anno_dict


def inter_rec(locxx, locyy):
    loc1 = locxx
    loc2 = locyy

    for i in range(0, len(locxx)):
        for j in range(0, len(locxx)):
            if i != j:
                Xmax = max(loc1[i][0], locxx[j][0])
                Ymax = max(loc1[i][1], locxx[j][1])
                M = (Xmax, Ymax)
                Xmin = min(loc2[i][0], locyy[j][0])
                Ymin = min(loc2[i][1], locyy[j][1])
                N = (Xmin, Ymin)
                if M[0] < N[0] and M[1] < N[1]: #判断矩形是否相交
                    loc1x = (min(loc1[i][0], locxx[j][0]), min(loc1[i][1], locxx[j][1]))
                    locly = (max(loc2[i][0], locyy[j][0]), max(loc2[i][1], locyy[j][1]))
                    aa = [loc1[i], loc1[j]]
                    bb = [loc2[i], loc2[j]]
                    loc1 = [loc1x if q in aa else q for q in loc1]
                    loc2 = [locly if w in bb else w for w in loc2]

    return loc1, loc2


def combined_union_bbox(locxx, locyy, margin=10):
    # locxx = [(0, 0), (478, 528), (185, 525), (423, 489), (200, 474), (595, 467), (488, 467), (313, 454), (391, 442),
    #          (244, 435),
    #          (240, 418), (431, 404), (437, 403), (352, 403), (365, 343), (303, 338), (436, 331), (343, 331), (222, 318),
    #          (353, 258), (241, 163)]
    # # 矩形右下角坐标：
    # locyy = [(400, 400), (494, 552), (201, 556), (485, 509), (222, 524), (613, 495), (544, 511), (340, 481), (423, 507),
    #          (308, 490),
    #          (264, 475), (468, 471), (459, 447), (385, 457), (382, 370), (335, 373), (459, 350), (365, 362), (294, 485),
    #          (391, 298), (542, 429)]

    locxx = [(i[0], i[1]) for i in locxx]    # 获取最开始每个HOI的坐标
    locyy = [(i[0], i[1]) for i in locyy]

    finx, finy = inter_rec(locxx, locyy)  # 进行第一次取union bbox的操作

    combin = []
    for k in range(len(finx)):
        for v in range(len(finy)):
            if k == v:
                combin.append(finx[k] + finy[v])
    combin = list(set(combin))
    # print('*****************************************************************************')
    # print(combin)
    # print('*****************************************************************************')

    locxx2 = [(i[0] - margin, i[1] - margin) for i in combin]
    locyy2 = [(i[2] + margin, i[3] + margin) for i in combin]

    finx2, finy2 = inter_rec(locxx2, locyy2)  # 进行第二次取union bbox的操作，即原来每个unionbox在打了padding之后如果和其他的HOI重叠了，我们就把这两个union box合成一个新的union box

    combin2 = []
    for k in range(len(finx2)):
        for v in range(len(finy2)):
            if k == v:
                combin2.append(finx2[k] + finy2[v])
    combin2 = list(set(combin2))   # 最终得到的union bbox就是padding之后合并重叠之后的union bbox，一张图片中可能有多个

    # Img = np.zeros([670, 700, 3], np.uint8) + 255
    #
    # # Img = cv2.copyMakeBorder(Img, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=[128, 128, 128])
    #
    # Img1 = Img.copy()
    # Img2 = Img.copy()
    # for i in range(0, len(locxx)):
    #     cv2.rectangle(Img, locxx[i], locyy[i], (0, 0, 255), 2)
    #
    # for j in range(0, len(combin)):
    #     cv2.rectangle(Img1, combin[j][0:2], combin[j][2:4], (255, 0, 0), 2)
    #
    # for j in range(0, len(combin2)):
    #     cv2.rectangle(Img2, combin2[j][0:2], combin2[j][2:4], (255, 0, 0), 2)
    #
    # cv2.imshow("img", Img)
    # cv2.imshow("result", Img1)
    # cv2.imshow("combined", Img2)
    # cv2.waitKey(0)

    return combin2


def get_union_bbox(random_index, annotations, dataset):

    anno_dict = get_basic_union_bbox(random_index, annotations, dataset)

    anno_file_name = anno_dict['file_name']
    anno_bbox = anno_dict['bbox']
    anno_id = anno_dict['bbox_id']
    hoi_sub = anno_dict['hoi_sub']
    hoi_obj = anno_dict['hoi_obj']
    verb_id = anno_dict['verb_id']
    hoi_id = anno_dict['hoi_id']
    hoi_union_bbox = anno_dict['hoi_union_bbox']

    # print('***************************************************************************')
    # print(anno_bbox)
    # print(anno_id)
    # print(hoi_sub)
    # print(hoi_obj)
    # print(hoi_id)
    # print(hoi_union_bbox)
    # print('***************************************************************************')

    proc_hoi_union_bbox = []
    proc_hoi_sub = []
    proc_hoi_obj = []
    proc_verb_id = []
    proc_hoi_id = []

    for i in range(len(random_index)):  # 提取原来每幅图上的所有单个HOI调用combined函数合并
        xymin = []
        xymax = []
        for j in range(len(hoi_union_bbox[i])):

            xymin.append(tuple(hoi_union_bbox[i][j][0:2]))
            xymax.append(tuple(hoi_union_bbox[i][j][2:4]))

        combin2 = combined_union_bbox(xymin, xymax)  # 得到这幅图合并之后的联合框
        # print('***************************************************')
        # print(combin2)
        # print('***************************************************')
        proc_hoi_union_bbox.append(combin2)

        hoi_sub_append = []
        hoi_obj_append = []
        verb_id_append = []
        hoi_id_append = []

        for p in range(len(proc_hoi_union_bbox[i])):  # 计算原来每个HOI分别属于现在合并后的哪个union bbox，因为我们后续
                                                        # 要随机选取unionbbox，需要知道这个unionbbox中有哪些HOI以及对应的人和物体的注释
            sub, obj, id1, id2 = [], [], [], []
            for q in range(len(hoi_union_bbox[i])):
                if hoi_union_bbox[i][q][0] >= proc_hoi_union_bbox[i][p][0] and hoi_union_bbox[i][q][1] >= proc_hoi_union_bbox[i][p][1] \
                and hoi_union_bbox[i][q][2] <= proc_hoi_union_bbox[i][p][2] and hoi_union_bbox[i][q][3] <= proc_hoi_union_bbox[i][p][3]:
                    sub.append(hoi_sub[i][q])
                    obj.append(hoi_obj[i][q])
                    id1.append(verb_id[i][q])
                    if dataset=='hico':
                        id2.append(hoi_id[i][q])
            hoi_sub_append.append(sub)  #每个union bbox中所有包含的sub合并起来
            hoi_obj_append.append(obj)   #每个union bbox中所有包含的obj合并起来，注意这里的顺序是对应的不会乱
            verb_id_append.append(id1)
            hoi_id_append.append(id2)

        proc_hoi_sub.append(hoi_sub_append)
        proc_hoi_obj.append(hoi_obj_append)
        proc_verb_id.append(verb_id_append)
        proc_hoi_id.append(hoi_id_append)

    # print("****************************")
    # print(anno_bbox)
    # print(anno_id)
    # print(proc_hoi_union_bbox)
    # print(proc_hoi_sub)
    # print(proc_hoi_obj)
    # print(proc_verb_id)
    # print(proc_hoi_id)
    # print('**********************************************')

    # Img = np.zeros([670, 700, 3], np.uint8) + 255
    # Img1 = Img.copy()
    #
    # # print(xymin)
    # for l in range(0, len(xymin)):
    #     cv2.rectangle(Img, xymin[l], xymax[l], (0, 0, 255), 2)
    #     cv2.imshow("img", Img)
    #
    # for l in range(len(proc_hoi_union_bbox[i])):
    #     cv2.rectangle(Img1, proc_hoi_union_bbox[i][l][0:2], proc_hoi_union_bbox[i][l][2:4], (0, 0, 255), 2)
    #     cv2.imshow("results", Img1)
    # cv2.waitKey(0)

    proc_anno = []
    for i in range(len(random_index)):
        anno = {}
        t = np.random.randint(len(proc_hoi_union_bbox[i])) # 由于上边算出了一副图中所有的union bbox，而我们只挑一个，因此还要在每张图里随机挑一个unionbbox
        anno['src_file_name'] = anno_file_name[i]
        anno['bbox'] = anno_bbox[i]
        anno['bbox_id'] = anno_id[i]
        anno['hoi_sub'] = proc_hoi_sub[i][t]
        anno['hoi_obj'] = proc_hoi_obj[i][t]
        anno['hoi_union_bbox'] = [int(v) for v in proc_hoi_union_bbox[i][t]]
        anno['verb_id'] = proc_verb_id[i][t]
        anno['hoi_id'] = proc_hoi_id[i][t]
        proc_anno.append(anno)  #  每次将一个union bbox的位置存起来方便后边对原图进行裁切，并且存储这个union_bbox中所有的hoi注释信息

    return proc_anno


def get_new_anno_and_image(sample_num, proc_anno, images_folder, margin=10):

    images = []
    for i in range(sample_num):
        #  更新打完margin之后的所有HOI和bbox的坐标
        proc_anno[i]['hoi_union_bbox'] = [v + margin for v in proc_anno[i]['hoi_union_bbox']]

        for j in range(len(proc_anno[i]['bbox'])):
            proc_anno[i]['bbox'][j] = [v + margin for v in proc_anno[i]['bbox'][j]]

        #  给源图像周围打上padding（margin），放置union box超出图像
        src_img = Image.open(os.path.join(images_folder, proc_anno[i]['src_file_name']))
        src_img = np.array(src_img)
        pad_img = cv2.copyMakeBorder(src_img, margin, margin, margin, margin, cv2.BORDER_CONSTANT,
                                     value=[128, 128, 128])
        #  从原图中将这个HOI的图像截出来
        xmin, ymin, xmax, ymax = proc_anno[i]['hoi_union_bbox']
        pts1 = np.float32([[xmin, ymin], [xmax, ymin], [xmin, ymax], [xmax, ymax]])
        pts2 = np.float32([[0, 0], [xmax - xmin, 0], [0, ymax - ymin], [xmax - xmin, ymax - ymin]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        image_output = cv2.warpPerspective(pad_img, matrix, (xmax - xmin, ymax - ymin))
        images.append(image_output)

        #  更新截取出来之后的图片的坐标（因为截取出来之后一个HOI的左上角坐标相当于原点坐标了）
        union_xmin, union_ymin = proc_anno[i]['hoi_union_bbox'][0], proc_anno[i]['hoi_union_bbox'][1]
        #  都减去HOI左上角的坐标即可完成更新
        proc_anno[i]['hoi_union_bbox'][0] = proc_anno[i]['hoi_union_bbox'][0] - union_xmin
        proc_anno[i]['hoi_union_bbox'][1] = proc_anno[i]['hoi_union_bbox'][1] - union_ymin
        proc_anno[i]['hoi_union_bbox'][2] = proc_anno[i]['hoi_union_bbox'][2] - union_xmin
        proc_anno[i]['hoi_union_bbox'][3] = proc_anno[i]['hoi_union_bbox'][3] - union_ymin

        for j in range(len(proc_anno[i]['bbox'])):
            proc_anno[i]['bbox'][j][0] = proc_anno[i]['bbox'][j][0] - union_xmin
            proc_anno[i]['bbox'][j][1] = proc_anno[i]['bbox'][j][1] - union_ymin
            proc_anno[i]['bbox'][j][2] = proc_anno[i]['bbox'][j][2] - union_xmin
            proc_anno[i]['bbox'][j][3] = proc_anno[i]['bbox'][j][3] - union_ymin
        # 此时其实proc_anno注释中的union_bbox已经没用了，因为我们已经得到了最终剪裁的图片
        # 需要注意的是，此时proc_anno中bbox还包含了这个HOI来源的那张图片上的所有bbox
    return proc_anno, images


def random_flip_horizontal(sample_num, anno, img, p=0.5):

    for i in range(sample_num):
        width = img[i].shape[1]
        if np.random.random() < p:
            if len(img[i].shape) == 3:
                img[i] = img[i][:, ::-1, :]
            elif len(img[i].shape) == 2:
                img[i] = img[i][:, ::-1]
            anno[i]['bbox'] = [[(width-v[2]), v[1], (width-v[0]), v[3]] for v in anno[i]['bbox']]

    return anno, img


def random_rescale(sample_num, anno, img):
    for i in range(sample_num):
        rescale_ratio = np.random.uniform(0.8, 1.25)
        img[i] = cv2.resize(img[i], None, fx=rescale_ratio, fy=rescale_ratio)
        anno[i]['bbox'] = [[v[0]*rescale_ratio, v[1]*rescale_ratio, v[2]*rescale_ratio,
                             v[3]*rescale_ratio] for v in anno[i]['bbox']]
        # hoi_union_box没用了
        anno[i]['hoi_union_bbox'] = [v*rescale_ratio for v in anno[i]['hoi_union_bbox']]

    return anno, img


def blending_c(matrix1: np.ndarray, matrix2: np.ndarray, blending_len):
    if matrix1.shape[0] != matrix2.shape[0]:
        raise Exception('blending size don\'t match')

    matrix1_start_index = matrix1.shape[1] - blending_len
    matrix2_end_index = blending_len
    blend_new = np.zeros((matrix1.shape[0], blending_len))
    for j in range(matrix1.shape[0]):
        blend1 = matrix1[j][matrix1_start_index:]
        blend2 = matrix2[j][0: matrix2_end_index]
        for i in range(len(blend1)):
            blend_new[j][i] = ((len(blend1) - i) / (len(blend1) + 1)) * blend1[i] + \
            (i + 1) / (len(blend1) + 1) * blend2[i]

    blend_matrix = np.c_[matrix1[:, 0:matrix1_start_index], blend_new]
    blend_matrix = np.c_[blend_matrix, matrix2[:, matrix2_end_index:]]

    return blend_matrix


def blending_r(matrix1: np.ndarray, matrix2: np.ndarray, blending_len):
    if matrix1.shape[1] != matrix2.shape[1]:
        raise Exception('blending size don\'t match')

    matrix1_start_index = matrix1.shape[0] - blending_len
    matrix2_end_index = blending_len
    blend_new = np.zeros((blending_len, matrix1.shape[1]))
    for j in range(matrix1.shape[1]):
        blend1 = matrix1[:, j][matrix1_start_index:]
        blend2 = matrix2[:, j][0: matrix2_end_index]
        for i in range(len(blend1)):
            blend_new[i][j] = ((len(blend1) - i) / (len(blend1) + 1)) * blend1[i] + \
            (i + 1) / (len(blend1) + 1) * blend2[i]

    blend_matrix = np.r_[matrix1[0:matrix1_start_index, :], blend_new]
    blend_matrix = np.r_[blend_matrix, matrix2[matrix2_end_index:, :]]

    return blend_matrix

# 这里获得的就是一张合成后的完整图片和注释，但这里的注释还不是hico的格式
# 可视化时给的参数就是这个函数返回的注释，转换完之后的注释无法可视化
def get_random_stitch_images(rescale_annos, rescale_images, images_folder):

    img = [Image.fromarray(v) for v in rescale_images]
    area = [im.size[0]*im.size[1] for im in img]
    area_ind = sorted(range(len(area)), key=lambda x: area[x], reverse=True)

    img = [img[v] for v in area_ind]

    rescale_annos = [rescale_annos[v] for v in area_ind]
    # print('sum_area:{}'.format(sum(area)))
    # for i, im in enumerate(img):
        # print('第{}张图片的尺寸是：{}'.format(i, (im.size[0], im.size[1])))

    overall_area = sum(area) * 2 * 1.21 * (np.random.random() * 0.4 + 0.6)
    a_value = np.sqrt(overall_area)
    a = np.random.randint(int(a_value * 0.75), int(a_value * 1.25) + 1)
    b = int(overall_area) // a
    a_m = float(a) / 1.1
    b_m = float(b) / 1.1
    # print('底片长宽分别为：{}'.format((a, b)))
    # room_state = np.zeros((a, b))
    # plt.imshow(room_state*255, cmap='gray')
    # plt.show()
    to_image = Image.new('RGB', (a, b), color='gray')
    # plt.imshow(to_image)
    # plt.show()
    center_co = []
    use_area_ind = []
    pos_min_x, pos_min_y, pos_max_x, pos_max_y = [], [], [], []
    for i in range(len(img)):
        w, h = img[i].size[0], img[i].size[1]
        # print('第{}张图片的size为:{}'.format(i, (x, y)))
        t = 0
        q = 1
        while(q==1 and t<10):
            pos_x = np.random.beta(0.5, 0.5) * (a - w)
            pos_y = np.random.beta(0.5, 0.5) * (b - h)

            center_x = pos_x + w / 2
            center_y = pos_y + h / 2

            for j in range(len(center_co)):
                if (abs(center_x - center_co[j][0]) <= (img[i].size[0] / 2 + img[j].size[0] / 2 + 1) and
                        abs(center_y - center_co[j][1]) <= (img[i].size[1] / 2 + img[j].size[1] / 2 + 1)):
                    break
            else:
                q = 0
            t = t + 1
        if q == 0:
            use_area_ind.append(i)
            center_co.append((center_x, center_y))
            pos_min_x.append(int(pos_x))
            pos_min_y.append(int(pos_y))
            pos_max_x.append(int(pos_x) + img[i].size[0])
            pos_max_y.append(int(pos_y) + img[i].size[1])
            # room_state[int(pos_x):int(pos_x + w), int(pos_y): int(pos_y + h)] = 1
            to_image.paste(img[i], (int(pos_x), int(pos_y)))
            rescale_annos[i]['bbox'] = [[v[0] + int(pos_x), v[1] + int(pos_y),
                                         v[2] + int(pos_x), v[3] + int(pos_y)] for v in rescale_annos[i]['bbox']]
            # plot_union_bbox(rescale_annos[i], to_image, 0)

    # 从原来的大图上裁剪下来内容主体部分，去除掉过多的背景
    xmin, ymin, xmax, ymax = min(pos_min_x), min(pos_min_y), max(pos_max_x), max(pos_max_y)
    pts1 = np.float32([[xmin, ymin], [xmax, ymin], [xmin, ymax], [xmax, ymax]])
    pts2 = np.float32([[0, 0], [xmax - xmin, 0], [0, ymax - ymin], [xmax - xmin, ymax - ymin]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    image_output = Image.fromarray(cv2.warpPerspective(np.array(to_image), matrix, (xmax - xmin, ymax - ymin)))

    # 由于裁切了内容且左上角不一定是从00开始的，因此要对坐标进行平移
    for i in range(len(img)):
        rescale_annos[i]['bbox'] = [[v[0] - xmin, v[1] - ymin,
                                     v[2] - xmin, v[3] - ymin] for v in rescale_annos[i]['bbox']]

    # 将图片缩放到合适的比例大小
    ratio = image_output.size[0] / image_output.size[1]
    src = []  # 源图像
    src_ratio = []  # 存储源图像的比例，后续要进行比对看和哪张图的原始比例最接近
    for i in range(len(img)):
        src.append(Image.open(os.path.join(images_folder, rescale_annos[i]['src_file_name'])))
        src_ratio.append(src[i].size[0] / src[i].size[1])
    eps = []
    for i in range(len(src_ratio)):
        eps.append(abs(ratio - src_ratio[i]))

    eps_index = sorted(range(len(eps)), key=lambda x: eps[x])[0]

    w_ratio = src[eps_index].size[0] / image_output.size[0]
    h_ratio = src[eps_index].size[1] / image_output.size[1]

    image_output = image_output.resize((src[eps_index].size[0], src[eps_index].size[1]))

    for i in range(len(img)):
        rescale_annos[i]['bbox'] = [[v[0] * w_ratio, v[1] * h_ratio, v[2] * w_ratio, v[3] * h_ratio] for v in rescale_annos[i]['bbox']]

    for i in range(len(use_area_ind)):
        if i==0:
            pass
        else:
            l = len(rescale_annos[use_area_ind[i-1]]['bbox'])
            rescale_annos[use_area_ind[i]]['hoi_sub'] = [v + l for v in rescale_annos[use_area_ind[i]]['hoi_sub']]
            rescale_annos[use_area_ind[i]]['hoi_obj'] = [v + l for v in rescale_annos[use_area_ind[i]]['hoi_obj']]
            rescale_annos[use_area_ind[i]]['bbox'] = rescale_annos[use_area_ind[i-1]]['bbox'] + rescale_annos[use_area_ind[i]]['bbox']
            rescale_annos[use_area_ind[i]]['bbox_id'] = rescale_annos[use_area_ind[i-1]]['bbox_id'] + rescale_annos[use_area_ind[i]]['bbox_id']
            rescale_annos[use_area_ind[i]]['hoi_sub'] = rescale_annos[use_area_ind[i-1]]['hoi_sub'] + rescale_annos[use_area_ind[i]]['hoi_sub']
            rescale_annos[use_area_ind[i]]['hoi_obj'] = rescale_annos[use_area_ind[i-1]]['hoi_obj'] + rescale_annos[use_area_ind[i]]['hoi_obj']
            rescale_annos[use_area_ind[i]]['verb_id'] = rescale_annos[use_area_ind[i-1]]['verb_id'] + rescale_annos[use_area_ind[i]]['verb_id']
            rescale_annos[use_area_ind[i]]['hoi_id'] = rescale_annos[use_area_ind[i-1]]['hoi_id'] + rescale_annos[use_area_ind[i]]['hoi_id']
            rescale_annos[use_area_ind[i]]['src_file_name'] = rescale_annos[use_area_ind[i-1]]['src_file_name'] + rescale_annos[use_area_ind[i]]['src_file_name']

    return rescale_annos[use_area_ind[-1]], image_output


def get_random_stitch_images_manual(rescale_annos, rescale_images, images_folder):
    to_image = Image.new('RGB', (3000, 3000), color='gray')

    img = [Image.fromarray(v) for v in rescale_images]

    if len(img) == 2:
        pos = [0] * 2
        img_ratio = [v.size[0]/v.size[1] for v in img]
        # 当两张图（HOI）竖着都很长时横着拼
        if img_ratio[0] < 0.5 and img_ratio[1] < 0.5:
            max_h_index = sorted(range(2), key=lambda x: img[x].size[1], reverse=True)
            to_image.paste(img[max_h_index[0]], (0, 0))
            pos[max_h_index[0]] = (0, 0)

            to_image.paste(img[max_h_index[1]], (img[max_h_index[0]].size[0], 0))
            pos[max_h_index[1]] = (img[max_h_index[0]].size[0], 0)
            rescale_annos[max_h_index[1]]['bbox'] = [[v[0] + img[max_h_index[0]].size[0], v[1],
                                                      v[2] + img[max_h_index[0]].size[0], v[3]] for v in
                                                     rescale_annos[max_h_index[1]]['bbox']]
        # 其他的就竖着拼
        else:
            max_w_index = sorted(range(2), key=lambda x: img[x].size[0], reverse=True)
            to_image.paste(img[max_w_index[0]], (0, 0))
            pos[max_w_index[0]] = (0, 0)

            to_image.paste(img[max_w_index[1]], (0, img[max_w_index[0]].size[1]))
            pos[max_w_index[1]] = (0, img[max_w_index[0]].size[1])
            rescale_annos[max_w_index[1]]['bbox'] = [[v[0], v[1] + img[max_w_index[0]].size[1],
                                                      v[2], v[3] + img[max_w_index[0]].size[1]] for v in
                                                     rescale_annos[max_w_index[1]]['bbox']]

    elif len(img) == 3:
        pos = [0] * 3
        max_w_index = sorted(range(3), key=lambda x: img[x].size[0], reverse=True)
        to_image.paste(img[max_w_index[0]], (0, 0))
        pos[max_w_index[0]] = (0, 0)

        to_image.paste(img[max_w_index[1]], (0, img[max_w_index[0]].size[1]))
        pos[max_w_index[1]] = (0, img[max_w_index[0]].size[1])
        rescale_annos[max_w_index[1]]['bbox'] = [[v[0], v[1] + img[max_w_index[0]].size[1],
                                                  v[2], v[3] + img[max_w_index[0]].size[1]] for v in
                                                 rescale_annos[max_w_index[1]]['bbox']]

        h_ratio = img[max_w_index[1]].size[1] / img[max_w_index[2]].size[1]
        scale_w = round(h_ratio * img[max_w_index[2]].size[0])
        scale_h = round(h_ratio * img[max_w_index[2]].size[1])
        img[max_w_index[2]] = img[max_w_index[2]].resize((scale_w, scale_h))
        to_image.paste(img[max_w_index[2]], (img[max_w_index[1]].size[0], img[max_w_index[0]].size[1]))
        pos[max_w_index[2]] = (img[max_w_index[1]].size[0], img[max_w_index[0]].size[1])
        rescale_annos[max_w_index[2]]['bbox'] = [[v[0] * h_ratio + img[max_w_index[1]].size[0],
                                                  v[1] * h_ratio + img[max_w_index[0]].size[1],
                                                  v[2] * h_ratio + img[max_w_index[1]].size[0],
                                                  v[3] * h_ratio + img[max_w_index[0]].size[1]] for v in rescale_annos[max_w_index[2]]['bbox']]

    elif len(img) == 4:
        pos = [0] * 4
        max_w_index = sorted(range(4), key=lambda x: img[x].size[0], reverse=True)
        to_image.paste(img[max_w_index[0]], (0, 0))
        pos[max_w_index[0]] = (0, 0)

        to_image.paste(img[max_w_index[3]], (img[max_w_index[0]].size[0], 0))
        pos[max_w_index[3]] = (img[max_w_index[0]].size[0], 0)
        rescale_annos[max_w_index[3]]['bbox'] = [[v[0] + img[max_w_index[0]].size[0], v[1],
                                                  v[2] + img[max_w_index[0]].size[0], v[3]] for v in
                                                 rescale_annos[max_w_index[3]]['bbox']]

        row1_h = max(img[max_w_index[0]].size[1], img[max_w_index[3]].size[1])

        to_image.paste(img[max_w_index[1]], (0, row1_h))
        pos[max_w_index[1]] = (0, row1_h)
        rescale_annos[max_w_index[1]]['bbox'] = [[v[0], v[1] + row1_h,
                                                  v[2], v[3] + row1_h] for v in
                                                  rescale_annos[max_w_index[1]]['bbox']]

        to_image.paste(img[max_w_index[2]], (img[max_w_index[1]].size[0], row1_h))
        pos[max_w_index[2]] = (img[max_w_index[1]].size[0], row1_h)
        rescale_annos[max_w_index[2]]['bbox'] = [[v[0] + img[max_w_index[1]].size[0], v[1] + row1_h,
                                                  v[2] + img[max_w_index[1]].size[0], v[3] + row1_h] for v in
                                                  rescale_annos[max_w_index[2]]['bbox']]

    elif len(img) == 5:
        pos = [0] * 5
        max_w_index = sorted(range(5), key=lambda x: img[x].size[0], reverse=True)
        to_image.paste(img[max_w_index[0]], (0, 0))
        pos[max_w_index[0]] = (0, 0)

        to_image.paste(img[max_w_index[1]], (img[max_w_index[0]].size[0], 0))
        pos[max_w_index[1]] = (img[max_w_index[0]].size[0], 0)
        rescale_annos[max_w_index[1]]['bbox'] = [[v[0] + img[max_w_index[0]].size[0], v[1],
                                                  v[2] + img[max_w_index[0]].size[0], v[3]] for
                                                 v in rescale_annos[max_w_index[1]]['bbox']]

        row1_h = max(img[max_w_index[0]].size[1], img[max_w_index[1]].size[1])


        to_image.paste(img[max_w_index[2]], (0, row1_h))
        pos[max_w_index[2]] = (0, row1_h)
        rescale_annos[max_w_index[2]]['bbox'] = [[v[0], v[1] + row1_h,
                                                  v[2], v[3] + row1_h] for v in
                                                 rescale_annos[max_w_index[2]]['bbox']]

        to_image.paste(img[max_w_index[3]], (img[max_w_index[2]].size[0], row1_h))
        pos[max_w_index[3]] = (img[max_w_index[2]].size[0], row1_h)
        rescale_annos[max_w_index[3]]['bbox'] = [[v[0] + img[max_w_index[2]].size[0],
                                                  v[1] + row1_h,
                                                  v[2] + img[max_w_index[2]].size[0],
                                                  v[3] + row1_h] for v in rescale_annos[max_w_index[3]]['bbox']]

        to_image.paste(img[max_w_index[4]], (img[max_w_index[2]].size[0] + img[max_w_index[3]].size[0]
                                             , row1_h))
        pos[max_w_index[4]] = (img[max_w_index[2]].size[0] + img[max_w_index[3]].size[0]
                                             , row1_h)
        rescale_annos[max_w_index[4]]['bbox'] = [[v[0] + img[max_w_index[2]].size[0] + img[max_w_index[3]].size[0],
                                                  v[1] + row1_h,
                                                  v[2] + img[max_w_index[2]].size[0] + img[max_w_index[3]].size[0],
                                                  v[3] + row1_h] for v in rescale_annos[max_w_index[4]]['bbox']]

    elif len(img) == 6:
        pos = [0] * 6
        max_w_index = sorted(range(6), key=lambda x: img[x].size[0], reverse=True)

        to_image.paste(img[max_w_index[0]], (0, 0))
        pos[max_w_index[0]] = (0, 0)

        to_image.paste(img[max_w_index[1]], (img[max_w_index[0]].size[0], 0))
        pos[max_w_index[1]] = (img[max_w_index[0]].size[0], 0)
        rescale_annos[max_w_index[1]]['bbox'] = [[v[0] + img[max_w_index[0]].size[0], v[1],
                                                  v[2] + img[max_w_index[0]].size[0], v[3]] for
                                                 v in rescale_annos[max_w_index[1]]['bbox']]

        to_image.paste(img[max_w_index[5]], (img[max_w_index[0]].size[0] + img[max_w_index[1]].size[0], 0))
        pos[max_w_index[5]] =  (img[max_w_index[0]].size[0] + img[max_w_index[1]].size[0], 0)
        rescale_annos[max_w_index[5]]['bbox'] = [[v[0] + img[max_w_index[0]].size[0] + img[max_w_index[1]].size[0],
                                                  v[1],
                                                  v[2] + img[max_w_index[0]].size[0] + img[max_w_index[1]].size[0],
                                                  v[3]] for v in rescale_annos[max_w_index[5]]['bbox']]

        row1_h = max(img[max_w_index[0]].size[1], img[max_w_index[1]].size[1], img[max_w_index[5]].size[1])

        to_image.paste(img[max_w_index[2]], (0, row1_h))
        pos[max_w_index[2]] = (0, row1_h)
        rescale_annos[max_w_index[2]]['bbox'] = [[v[0], v[1] + row1_h,
                                                  v[2], v[3] + row1_h] for v in
                                                 rescale_annos[max_w_index[2]]['bbox']]

        to_image.paste(img[max_w_index[3]], (img[max_w_index[2]].size[0], row1_h))
        pos[max_w_index[3]] = (img[max_w_index[2]].size[0], row1_h)
        rescale_annos[max_w_index[3]]['bbox'] = [[v[0] + img[max_w_index[2]].size[0],
                                                  v[1] + row1_h,
                                                  v[2] + img[max_w_index[2]].size[0],
                                                  v[3] + row1_h] for v in rescale_annos[max_w_index[3]]['bbox']]

        to_image.paste(img[max_w_index[4]], (img[max_w_index[2]].size[0] + img[max_w_index[3]].size[0]
                                             , row1_h))
        pos[max_w_index[4]] = (img[max_w_index[2]].size[0] + img[max_w_index[3]].size[0]
                               , row1_h)
        rescale_annos[max_w_index[4]]['bbox'] = [[v[0] + img[max_w_index[2]].size[0] + img[max_w_index[3]].size[0],
                                                  v[1] + row1_h,
                                                  v[2] + img[max_w_index[2]].size[0] + img[max_w_index[3]].size[0],
                                                  v[3] + row1_h] for v in rescale_annos[max_w_index[4]]['bbox']]

    elif len(img) == 7:
        pos = [0] * 7
        max_w_index = sorted(range(7), key=lambda x: img[x].size[0], reverse=True)

        to_image.paste(img[max_w_index[0]], (0, 0))
        pos[max_w_index[0]] = (0, 0)

        to_image.paste(img[max_w_index[1]], (img[max_w_index[0]].size[0], 0))
        pos[max_w_index[1]] = (img[max_w_index[0]].size[0], 0)
        rescale_annos[max_w_index[1]]['bbox'] = [[v[0] + img[max_w_index[0]].size[0], v[1],
                                                  v[2] + img[max_w_index[0]].size[0], v[3]] for
                                                 v in rescale_annos[max_w_index[1]]['bbox']]

        to_image.paste(img[max_w_index[6]], (img[max_w_index[0]].size[0] + img[max_w_index[1]].size[0], 0))
        pos[max_w_index[6]] = (img[max_w_index[0]].size[0] + img[max_w_index[1]].size[0], 0)
        rescale_annos[max_w_index[6]]['bbox'] = [[v[0] + img[max_w_index[0]].size[0] + img[max_w_index[1]].size[0],
                                                  v[1],
                                                  v[2] + img[max_w_index[0]].size[0] + img[max_w_index[1]].size[0],
                                                  v[3]] for v in rescale_annos[max_w_index[6]]['bbox']]

        row1_h = max(img[max_w_index[0]].size[1], img[max_w_index[1]].size[1], img[max_w_index[6]].size[1])

        to_image.paste(img[max_w_index[2]], (0, row1_h))
        pos[max_w_index[2]] = (0, row1_h)
        rescale_annos[max_w_index[2]]['bbox'] = [[v[0], v[1] + row1_h,
                                                  v[2], v[3] + row1_h] for v in
                                                 rescale_annos[max_w_index[2]]['bbox']]

        to_image.paste(img[max_w_index[3]], (img[max_w_index[2]].size[0], row1_h))
        pos[max_w_index[3]] = (img[max_w_index[2]].size[0], row1_h)
        rescale_annos[max_w_index[3]]['bbox'] = [[v[0] + img[max_w_index[2]].size[0],
                                                  v[1] + row1_h,
                                                  v[2] + img[max_w_index[2]].size[0],
                                                  v[3] + row1_h] for v in rescale_annos[max_w_index[3]]['bbox']]

        to_image.paste(img[max_w_index[5]], (img[max_w_index[2]].size[0] + img[max_w_index[3]].size[0]
                                             , row1_h))
        pos[max_w_index[5]] = (img[max_w_index[2]].size[0] + img[max_w_index[3]].size[0]
                               , row1_h)
        rescale_annos[max_w_index[5]]['bbox'] = [[v[0] + img[max_w_index[2]].size[0] + img[max_w_index[3]].size[0],
                                                  v[1] + row1_h,
                                                  v[2] + img[max_w_index[2]].size[0] + img[max_w_index[3]].size[0],
                                                  v[3] + row1_h] for v in rescale_annos[max_w_index[5]]['bbox']]

        row2_h = max(img[max_w_index[2]].size[1], img[max_w_index[3]].size[1], img[max_w_index[5]].size[1]) + row1_h

        to_image.paste(img[max_w_index[4]], (0, row2_h))
        pos[max_w_index[4]] = (0, row2_h)
        rescale_annos[max_w_index[4]]['bbox'] = [[v[0], v[1] + row2_h,
                                                  v[2], v[3] + row2_h] for v in
                                                 rescale_annos[max_w_index[4]]['bbox']]

    elif len(img) == 8:
        pos = [0] * 8
        max_w_index = sorted(range(8), key=lambda x: img[x].size[0], reverse=True)

        to_image.paste(img[max_w_index[0]], (0, 0))
        pos[max_w_index[0]] = (0, 0)

        to_image.paste(img[max_w_index[1]], (img[max_w_index[0]].size[0], 0))
        pos[max_w_index[1]] = (img[max_w_index[0]].size[0], 0)
        rescale_annos[max_w_index[1]]['bbox'] = [[v[0] + img[max_w_index[0]].size[0], v[1],
                                                  v[2] + img[max_w_index[0]].size[0], v[3]] for
                                                 v in rescale_annos[max_w_index[1]]['bbox']]

        to_image.paste(img[max_w_index[7]], (img[max_w_index[0]].size[0] + img[max_w_index[1]].size[0], 0))
        pos[max_w_index[7]] = (img[max_w_index[0]].size[0] + img[max_w_index[1]].size[0], 0)
        rescale_annos[max_w_index[7]]['bbox'] = [[v[0] + img[max_w_index[0]].size[0] + img[max_w_index[1]].size[0],
                                                  v[1],
                                                  v[2] + img[max_w_index[0]].size[0] + img[max_w_index[1]].size[0],
                                                  v[3]] for v in rescale_annos[max_w_index[7]]['bbox']]

        row1_h = max(img[max_w_index[0]].size[1], img[max_w_index[1]].size[1], img[max_w_index[7]].size[1])

        to_image.paste(img[max_w_index[2]], (0, row1_h))
        pos[max_w_index[2]] = (0, row1_h)
        rescale_annos[max_w_index[2]]['bbox'] = [[v[0], v[1] + row1_h,
                                                  v[2], v[3] + row1_h] for v in
                                                 rescale_annos[max_w_index[2]]['bbox']]

        to_image.paste(img[max_w_index[3]], (img[max_w_index[2]].size[0], row1_h))
        pos[max_w_index[3]] = (img[max_w_index[2]].size[0], row1_h)
        rescale_annos[max_w_index[3]]['bbox'] = [[v[0] + img[max_w_index[2]].size[0],
                                                  v[1] + row1_h,
                                                  v[2] + img[max_w_index[2]].size[0],
                                                  v[3] + row1_h] for v in rescale_annos[max_w_index[3]]['bbox']]

        to_image.paste(img[max_w_index[6]], (img[max_w_index[2]].size[0] + img[max_w_index[3]].size[0]
                                             , row1_h))
        pos[max_w_index[6]] = (img[max_w_index[2]].size[0] + img[max_w_index[3]].size[0]
                               , row1_h)
        rescale_annos[max_w_index[6]]['bbox'] = [[v[0] + img[max_w_index[2]].size[0] + img[max_w_index[3]].size[0],
                                                  v[1] + row1_h,
                                                  v[2] + img[max_w_index[2]].size[0] + img[max_w_index[3]].size[0],
                                                  v[3] + row1_h] for v in rescale_annos[max_w_index[6]]['bbox']]

        row2_h = max(img[max_w_index[2]].size[1], img[max_w_index[3]].size[1], img[max_w_index[6]].size[1]) + row1_h

        to_image.paste(img[max_w_index[4]], (0, row2_h))
        pos[max_w_index[4]] = (0, row2_h)
        rescale_annos[max_w_index[4]]['bbox'] = [[v[0], v[1] + row2_h,
                                                  v[2], v[3] + row2_h] for v in
                                                 rescale_annos[max_w_index[4]]['bbox']]

        to_image.paste(img[max_w_index[5]], (img[max_w_index[4]].size[0], row2_h))
        pos[max_w_index[5]] = (img[max_w_index[4]].size[0], row2_h)
        rescale_annos[max_w_index[5]]['bbox'] = [[v[0] + img[max_w_index[4]].size[0], v[1] + row2_h,
                                                  v[2] + img[max_w_index[4]].size[0], v[3] + row2_h] for v in
                                                 rescale_annos[max_w_index[5]]['bbox']]

    elif len(img) == 9:
        pos = [0] * 9
        max_w_index = sorted(range(9), key=lambda x: img[x].size[0], reverse=True)

        to_image.paste(img[max_w_index[0]], (0, 0))
        pos[max_w_index[0]] = (0, 0)

        to_image.paste(img[max_w_index[1]], (img[max_w_index[0]].size[0], 0))
        pos[max_w_index[1]] = (img[max_w_index[0]].size[0], 0)
        rescale_annos[max_w_index[1]]['bbox'] = [[v[0] + img[max_w_index[0]].size[0], v[1],
                                                  v[2] + img[max_w_index[0]].size[0], v[3]] for
                                                 v in rescale_annos[max_w_index[1]]['bbox']]

        to_image.paste(img[max_w_index[8]], (img[max_w_index[0]].size[0] + img[max_w_index[1]].size[0], 0))
        pos[max_w_index[8]] = (img[max_w_index[0]].size[0] + img[max_w_index[1]].size[0], 0)
        rescale_annos[max_w_index[8]]['bbox'] = [[v[0] + img[max_w_index[0]].size[0] + img[max_w_index[1]].size[0],
                                                  v[1],
                                                  v[2] + img[max_w_index[0]].size[0] + img[max_w_index[1]].size[0],
                                                  v[3]] for v in rescale_annos[max_w_index[8]]['bbox']]

        row1_h = max(img[max_w_index[0]].size[1], img[max_w_index[1]].size[1], img[max_w_index[8]].size[1])

        to_image.paste(img[max_w_index[2]], (0, row1_h))
        pos[max_w_index[2]] = (0, row1_h)
        rescale_annos[max_w_index[2]]['bbox'] = [[v[0], v[1] + row1_h,
                                                  v[2], v[3] + row1_h] for v in
                                                 rescale_annos[max_w_index[2]]['bbox']]

        to_image.paste(img[max_w_index[3]], (img[max_w_index[2]].size[0], row1_h))
        pos[max_w_index[3]] = (img[max_w_index[2]].size[0], row1_h)
        rescale_annos[max_w_index[3]]['bbox'] = [[v[0] + img[max_w_index[2]].size[0],
                                                  v[1] + row1_h,
                                                  v[2] + img[max_w_index[2]].size[0],
                                                  v[3] + row1_h] for v in rescale_annos[max_w_index[3]]['bbox']]

        to_image.paste(img[max_w_index[7]], (img[max_w_index[2]].size[0] + img[max_w_index[3]].size[0]
                                             , row1_h))
        pos[max_w_index[7]] = (img[max_w_index[2]].size[0] + img[max_w_index[3]].size[0]
                               , row1_h)
        rescale_annos[max_w_index[7]]['bbox'] = [[v[0] + img[max_w_index[2]].size[0] + img[max_w_index[3]].size[0],
                                                  v[1] + row1_h,
                                                  v[2] + img[max_w_index[2]].size[0] + img[max_w_index[3]].size[0],
                                                  v[3] + row1_h] for v in rescale_annos[max_w_index[7]]['bbox']]

        row2_h = max(img[max_w_index[2]].size[1], img[max_w_index[3]].size[1], img[max_w_index[7]].size[1]) + row1_h

        to_image.paste(img[max_w_index[4]], (0, row2_h))
        pos[max_w_index[4]] = (0, row2_h)
        rescale_annos[max_w_index[4]]['bbox'] = [[v[0], v[1] + row2_h,
                                                  v[2], v[3] + row2_h] for v in
                                                 rescale_annos[max_w_index[4]]['bbox']]

        to_image.paste(img[max_w_index[5]], (img[max_w_index[4]].size[0], row2_h))
        pos[max_w_index[5]] = (img[max_w_index[4]].size[0], row2_h)
        rescale_annos[max_w_index[5]]['bbox'] = [[v[0] + img[max_w_index[4]].size[0], v[1] + row2_h,
                                                  v[2] + img[max_w_index[4]].size[0], v[3] + row2_h] for v in
                                                 rescale_annos[max_w_index[5]]['bbox']]

        to_image.paste(img[max_w_index[6]], (img[max_w_index[4]].size[0] + img[max_w_index[5]].size[0], row2_h))
        pos[max_w_index[6]] = (img[max_w_index[4]].size[0] + img[max_w_index[5]].size[0], row2_h)
        rescale_annos[max_w_index[6]]['bbox'] = [[v[0] + img[max_w_index[4]].size[0] + img[max_w_index[5]].size[0],
                                                  v[1] + row2_h,
                                                  v[2] + img[max_w_index[4]].size[0] + img[max_w_index[5]].size[0],
                                                  v[3] + row2_h] for v in
                                                 rescale_annos[max_w_index[6]]['bbox']]

    elif len(img) == 10:
        pos = [0] * 10
        max_w_index = sorted(range(10), key=lambda x: img[x].size[0], reverse=True)

        to_image.paste(img[max_w_index[0]], (0, 0))
        pos[max_w_index[0]] = (0, 0)

        to_image.paste(img[max_w_index[1]], (img[max_w_index[0]].size[0], 0))
        pos[max_w_index[1]] = (img[max_w_index[0]].size[0], 0)
        rescale_annos[max_w_index[1]]['bbox'] = [[v[0] + img[max_w_index[0]].size[0], v[1],
                                                  v[2] + img[max_w_index[0]].size[0], v[3]] for
                                                 v in rescale_annos[max_w_index[1]]['bbox']]

        to_image.paste(img[max_w_index[8]], (img[max_w_index[0]].size[0] + img[max_w_index[1]].size[0], 0))
        pos[max_w_index[8]] = (img[max_w_index[0]].size[0] + img[max_w_index[1]].size[0], 0)
        rescale_annos[max_w_index[8]]['bbox'] = [[v[0] + img[max_w_index[0]].size[0] + img[max_w_index[1]].size[0],
                                                  v[1],
                                                  v[2] + img[max_w_index[0]].size[0] + img[max_w_index[1]].size[0],
                                                  v[3]] for v in rescale_annos[max_w_index[8]]['bbox']]

        row1_h = max(img[max_w_index[0]].size[1], img[max_w_index[1]].size[1], img[max_w_index[8]].size[1])

        to_image.paste(img[max_w_index[2]], (0, row1_h))
        pos[max_w_index[2]] = (0, row1_h)
        rescale_annos[max_w_index[2]]['bbox'] = [[v[0], v[1] + row1_h,
                                                  v[2], v[3] + row1_h] for v in
                                                 rescale_annos[max_w_index[2]]['bbox']]

        to_image.paste(img[max_w_index[3]], (img[max_w_index[2]].size[0], row1_h))
        pos[max_w_index[3]] = (img[max_w_index[2]].size[0], row1_h)
        rescale_annos[max_w_index[3]]['bbox'] = [[v[0] + img[max_w_index[2]].size[0],
                                                  v[1] + row1_h,
                                                  v[2] + img[max_w_index[2]].size[0],
                                                  v[3] + row1_h] for v in rescale_annos[max_w_index[3]]['bbox']]

        to_image.paste(img[max_w_index[7]], (img[max_w_index[2]].size[0] + img[max_w_index[3]].size[0]
                                             , row1_h))
        pos[max_w_index[7]] = (img[max_w_index[2]].size[0] + img[max_w_index[3]].size[0]
                               , row1_h)
        rescale_annos[max_w_index[7]]['bbox'] = [[v[0] + img[max_w_index[2]].size[0] + img[max_w_index[3]].size[0],
                                                  v[1] + row1_h,
                                                  v[2] + img[max_w_index[2]].size[0] + img[max_w_index[3]].size[0],
                                                  v[3] + row1_h] for v in rescale_annos[max_w_index[7]]['bbox']]

        row2_h = max(img[max_w_index[2]].size[1], img[max_w_index[3]].size[1], img[max_w_index[7]].size[1]) + row1_h

        to_image.paste(img[max_w_index[4]], (0, row2_h))
        pos[max_w_index[4]] = (0, row2_h)
        rescale_annos[max_w_index[4]]['bbox'] = [[v[0], v[1] + row2_h,
                                                  v[2], v[3] + row2_h] for v in
                                                 rescale_annos[max_w_index[4]]['bbox']]

        to_image.paste(img[max_w_index[5]], (img[max_w_index[4]].size[0], row2_h))
        pos[max_w_index[5]] = (img[max_w_index[4]].size[0], row2_h)
        rescale_annos[max_w_index[5]]['bbox'] = [[v[0] + img[max_w_index[4]].size[0], v[1] + row2_h,
                                                  v[2] + img[max_w_index[4]].size[0], v[3] + row2_h] for v in
                                                 rescale_annos[max_w_index[5]]['bbox']]

        to_image.paste(img[max_w_index[6]], (img[max_w_index[4]].size[0] + img[max_w_index[5]].size[0], row2_h))
        pos[max_w_index[6]] = (img[max_w_index[4]].size[0] + img[max_w_index[5]].size[0], row2_h)
        rescale_annos[max_w_index[6]]['bbox'] = [[v[0] + img[max_w_index[4]].size[0] + img[max_w_index[5]].size[0],
                                                  v[1] + row2_h,
                                                  v[2] + img[max_w_index[4]].size[0] + img[max_w_index[5]].size[0],
                                                  v[3] + row2_h] for v in
                                                 rescale_annos[max_w_index[6]]['bbox']]

        to_image.paste(img[max_w_index[9]], (img[max_w_index[4]].size[0] + img[max_w_index[5]].size[0] + img[max_w_index[6]].size[0], row2_h))
        pos[max_w_index[9]] = (img[max_w_index[4]].size[0] + img[max_w_index[5]].size[0] + img[max_w_index[6]].size[0], row2_h)
        rescale_annos[max_w_index[9]]['bbox'] = [[v[0] + img[max_w_index[4]].size[0] + img[max_w_index[5]].size[0] + img[max_w_index[6]].size[0],
                                                  v[1] + row2_h,
                                                  v[2] + img[max_w_index[4]].size[0] + img[max_w_index[5]].size[0] + img[max_w_index[6]].size[0],
                                                  v[3] + row2_h] for v in
                                                 rescale_annos[max_w_index[9]]['bbox']]

    else:
        raise Exception('not support images_num')

    # 获得拼好的图片的左上右下坐标，从背板中裁剪下来
    pos_min_x, pos_min_y, pos_max_x, pos_max_y = [], [], [], []
    for i in range(len(img)):
        pos_min_x.append(pos[i][0])
        pos_min_y.append(pos[i][1])
        pos_max_x.append(pos[i][0] + img[i].size[0])
        pos_max_y.append(pos[i][1] + img[i].size[1])

    xmin, ymin, xmax, ymax = min(pos_min_x), min(pos_min_y), max(pos_max_x), max(pos_max_y)
    pts1 = np.float32([[xmin, ymin], [xmax, ymin], [xmin, ymax], [xmax, ymax]])
    pts2 = np.float32([[0, 0], [xmax - xmin, 0], [0, ymax - ymin], [xmax - xmin, ymax - ymin]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    image_output = Image.fromarray(cv2.warpPerspective(np.array(to_image), matrix, (xmax - xmin, ymax - ymin)))

    # 将图片缩放到合适的比例大小
    ratio = image_output.size[0] / image_output.size[1]
    src = []  # 源图像
    src_ratio = []  # 存储源图像的比例，后续要进行比对看和哪张图的原始比例最接近
    for i in range(len(img)):
        src.append(Image.open(os.path.join(images_folder, rescale_annos[i]['src_file_name'])))
        src_ratio.append(src[i].size[0] / src[i].size[1])
    eps = []
    for i in range(len(src_ratio)):
        eps.append(abs(ratio - src_ratio[i]))

    eps_index = sorted(range(len(eps)), key=lambda x: eps[x])[0]

    if eps[eps_index] < 0.5:
        w_ratio = src[eps_index].size[0] / image_output.size[0]
        h_ratio = src[eps_index].size[1] / image_output.size[1]
        image_output = image_output.resize((src[eps_index].size[0], src[eps_index].size[1]))
        for i in range(len(img)):
            rescale_annos[i]['bbox'] = [[v[0] * w_ratio, v[1] * h_ratio, v[2] * w_ratio, v[3] * h_ratio] for v in
                                        rescale_annos[i]['bbox']]
    else:
        w_ratio = src[eps_index].size[0] / image_output.size[0]
        new_h = round(w_ratio * image_output.size[1])
        image_output = image_output.resize((src[eps_index].size[0], new_h))
        for i in range(len(img)):
            rescale_annos[i]['bbox'] = [[v[0] * w_ratio, v[1] * w_ratio, v[2] * w_ratio, v[3] * w_ratio] for v in
                                        rescale_annos[i]['bbox']]

    for i in range(len(img)):
        if i==0:
            pass
        else:
            l = len(rescale_annos[i-1]['bbox'])
            rescale_annos[i]['hoi_sub'] = [v + l for v in rescale_annos[i]['hoi_sub']]
            rescale_annos[i]['hoi_obj'] = [v + l for v in rescale_annos[i]['hoi_obj']]
            rescale_annos[i]['bbox'] = rescale_annos[i-1]['bbox'] + rescale_annos[i]['bbox']
            rescale_annos[i]['bbox_id'] = rescale_annos[i - 1]['bbox_id'] + rescale_annos[i]['bbox_id']
            rescale_annos[i]['hoi_sub'] = rescale_annos[i - 1]['hoi_sub'] + rescale_annos[i]['hoi_sub']
            rescale_annos[i]['hoi_obj'] = rescale_annos[i - 1]['hoi_obj'] + rescale_annos[i]['hoi_obj']
            rescale_annos[i]['verb_id'] = rescale_annos[i - 1]['verb_id'] + rescale_annos[i]['verb_id']
            rescale_annos[i]['hoi_id'] = rescale_annos[i - 1]['hoi_id'] + rescale_annos[i]['hoi_id']
            rescale_annos[i]['src_file_name'] = rescale_annos[i - 1]['src_file_name'] + rescale_annos[i]['src_file_name']

    return rescale_annos[len(rescale_images)-1], image_output


def convanno2hico(anno):
    new_annotations = {}
    new_annotations_annotations = []

    for i in range(len(anno['bbox'])):
        d = {}
        d['bbox'] = anno['bbox'][i]
        d['category_id'] = anno['bbox_id'][i]
        new_annotations_annotations.append(d)
    new_annotations['annotations'] = new_annotations_annotations

    new_annotations_hoi_annotation = []
    for i in range(len(anno['hoi_sub'])):
        d = {}
        d['subject_id'] = anno['hoi_sub'][i]
        d['object_id'] = anno['hoi_obj'][i]
        d['category_id'] = anno['verb_id'][i]
        d['hoi_category_id'] = anno['hoi_id'][i]
        new_annotations_hoi_annotation.append(d)
    new_annotations['hoi_annotation'] = new_annotations_hoi_annotation
    return new_annotations


def convanno2vcoco(anno):

    new_annotations = {}
    new_annotations_annotations = []

    for i in range(len(anno['bbox'])):
        d = {}
        d['bbox'] = anno['bbox'][i]
        d['category_id'] = anno['bbox_id'][i]
        new_annotations_annotations.append(d)
    new_annotations['annotations'] = new_annotations_annotations

    new_annotations_hoi_annotation = []
    for i in range(len(anno['hoi_sub'])):
        d = {}
        d['subject_id'] = anno['hoi_sub'][i]
        d['object_id'] = anno['hoi_obj'][i]
        d['category_id'] = anno['verb_id'][i]
        new_annotations_hoi_annotation.append(d)
    new_annotations['hoi_annotation'] = new_annotations_hoi_annotation
    return new_annotations


def get_replace_image(random_index, annotations, images_folder, dataset_file):

    proc_anno = get_union_bbox(random_index, annotations, dataset_file)
    new_annos, new_imgaes = get_new_anno_and_image(len(random_index), proc_anno, images_folder)
    flip_annos, flip_images = random_flip_horizontal(len(random_index), new_annos, new_imgaes)
    rescale_annos, rescale_images = random_rescale(len(random_index), flip_annos, flip_images)
    anno, image = get_random_stitch_images_manual(rescale_annos, rescale_images, images_folder)
    # plot_union_bbox(anno, image, 0)
    if dataset_file == 'vcoco':
        anno = convanno2vcoco(anno)
    elif dataset_file == 'hico':
        anno = convanno2hico(anno)
    elif dataset_file == 'hoia':
        anno = convanno2vcoco(anno)
    else:
        raise Exception('not support dateset')
    return anno, image


#########单独测试blending有没有用###########################
def get_blending_images4(rescale_annos, rescale_images, images_folder):
    img = [Image.fromarray(v).convert('RGB') for v in rescale_images]

    max_w_index = sorted(range(4), key=lambda x: img[x].size[0], reverse=True)
    img_row1 = [img[max_w_index[0]], img[max_w_index[3]]]
    rescale_annos_row1 = [rescale_annos[max_w_index[0]], rescale_annos[max_w_index[3]]]
    rescale_annos_row2 = [rescale_annos[max_w_index[1]], rescale_annos[max_w_index[2]]]
    max_h_index1 = sorted(range(2), key=lambda x: img_row1[x].size[1], reverse=True)

    fill_color = Image.new('RGB', (img_row1[max_h_index1[1]].size[0], img_row1[max_h_index1[0]].size[1] - img_row1[max_h_index1[1]].size[1] + 10), color='gray')

    blend_c1 = np.uint8(blending_r(np.array(img_row1[max_h_index1[1]])[:, :, 0], np.array(fill_color)[:, :, 0], 10))
    blend_c2 = np.uint8(blending_r(np.array(img_row1[max_h_index1[1]])[:, :, 1], np.array(fill_color)[:, :, 1], 10))
    blend_c3 = np.uint8(blending_r(np.array(img_row1[max_h_index1[1]])[:, :, 2], np.array(fill_color)[:, :, 2], 10))
    img_row1[max_h_index1[1]] = Image.fromarray(np.stack((blend_c1, blend_c2, blend_c3), axis=2))

    blend_c1 = np.uint8(blending_c(np.array(img_row1[max_h_index1[0]])[:, :, 0], np.array(img_row1[max_h_index1[1]])[:, :, 0], 10))
    blend_c2 = np.uint8(blending_c(np.array(img_row1[max_h_index1[0]])[:, :, 1], np.array(img_row1[max_h_index1[1]])[:, :, 1], 10))
    blend_c3 = np.uint8(blending_c(np.array(img_row1[max_h_index1[0]])[:, :, 2], np.array(img_row1[max_h_index1[1]])[:, :, 2], 10))

    image_out1 = Image.fromarray(np.stack((blend_c1, blend_c2, blend_c3), axis=2))

    rescale_annos_row1[max_h_index1[1]]['bbox'] = [[v[0] + img_row1[max_h_index1[0]].size[0] - 10, v[1],
                                                    v[2] + img_row1[max_h_index1[0]].size[0] - 10, v[3]] for v in
                                             rescale_annos_row1[max_h_index1[1]]['bbox']]

    img_row2 = [img[max_w_index[1]], img[max_w_index[2]]]

    max_h_index2 = sorted(range(2), key=lambda x: img_row2[x].size[1], reverse=True)

    fill_color = Image.new('RGB', (img_row2[max_h_index2[1]].size[0], img_row2[max_h_index2[0]].size[1] - img_row2[max_h_index2[1]].size[1] + 10),
                           color='gray')

    blend_c1 = np.uint8(blending_r(np.array(img_row2[max_h_index2[1]])[:, :, 0], np.array(fill_color)[:, :, 0], 10))
    blend_c2 = np.uint8(blending_r(np.array(img_row2[max_h_index2[1]])[:, :, 1], np.array(fill_color)[:, :, 1], 10))
    blend_c3 = np.uint8(blending_r(np.array(img_row2[max_h_index2[1]])[:, :, 2], np.array(fill_color)[:, :, 2], 10))
    img_row2[max_h_index2[1]] = Image.fromarray(np.stack((blend_c1, blend_c2, blend_c3), axis=2))

    blend_c1 = np.uint8(
        blending_c(np.array(img_row2[max_h_index2[0]])[:, :, 0], np.array(img_row2[max_h_index2[1]])[:, :, 0], 10))
    blend_c2 = np.uint8(
        blending_c(np.array(img_row2[max_h_index2[0]])[:, :, 1], np.array(img_row2[max_h_index2[1]])[:, :, 1], 10))
    blend_c3 = np.uint8(
        blending_c(np.array(img_row2[max_h_index2[0]])[:, :, 2], np.array(img_row2[max_h_index2[1]])[:, :, 2], 10))

    image_out2 = Image.fromarray(np.stack((blend_c1, blend_c2, blend_c3), axis=2))

    rescale_annos_row2[max_h_index2[1]]['bbox'] = [[v[0] + img_row2[max_h_index2[0]].size[0] - 10, v[1],
                                                    v[2] + img_row2[max_h_index2[0]].size[0] - 10, v[3]] for v in
                                                   rescale_annos_row2[max_h_index2[1]]['bbox']]

    img_stitch = [image_out1, image_out2]
    rescale_annos_stitch = [rescale_annos_row1, rescale_annos_row2]
    w_index_stitch = sorted(range(2), key=lambda x: img_stitch[x].size[0], reverse=True)

    fill_color = Image.new('RGB', (img_stitch[w_index_stitch[0]].size[0] - img_stitch[w_index_stitch[1]].size[0] + 10, img_stitch[w_index_stitch[1]].size[1]),
                           color='gray')

    blend_c1 = np.uint8(blending_c(np.array(img_stitch[w_index_stitch[1]])[:, :, 0], np.array(fill_color)[:, :, 0], 10))
    blend_c2 = np.uint8(blending_c(np.array(img_stitch[w_index_stitch[1]])[:, :, 1], np.array(fill_color)[:, :, 1], 10))
    blend_c3 = np.uint8(blending_c(np.array(img_stitch[w_index_stitch[1]])[:, :, 2], np.array(fill_color)[:, :, 2], 10))
    img_stitch[w_index_stitch[1]] = Image.fromarray(np.stack((blend_c1, blend_c2, blend_c3), axis=2))

    blend_c1 = np.uint8(
        blending_r(np.array(img_stitch[w_index_stitch[0]])[:, :, 0], np.array(img_stitch[w_index_stitch[1]])[:, :, 0], 10))
    blend_c2 = np.uint8(
        blending_r(np.array(img_stitch[w_index_stitch[0]])[:, :, 1], np.array(img_stitch[w_index_stitch[1]])[:, :, 1], 10))
    blend_c3 = np.uint8(
        blending_r(np.array(img_stitch[w_index_stitch[0]])[:, :, 2], np.array(img_stitch[w_index_stitch[1]])[:, :, 2], 10))

    image_output = Image.fromarray(np.stack((blend_c1, blend_c2, blend_c3), axis=2))

    h = img_stitch[w_index_stitch[0]].size[1]

    rescale_annos_stitch[w_index_stitch[1]][0]['bbox'] = [[v[0], v[1] + h - 10,
                                                           v[2], v[3] + h - 10] for v in
                                                   rescale_annos_stitch[w_index_stitch[1]][0]['bbox']]

    rescale_annos_stitch[w_index_stitch[1]][1]['bbox'] = [[v[0], v[1] + h - 10,
                                                           v[2], v[3] + h - 10] for v in
                                                          rescale_annos_stitch[w_index_stitch[1]][1]['bbox']]

    rescale_annos = rescale_annos_stitch[0] + rescale_annos_stitch[1]

    # 将图片缩放到合适的比例大小
    ratio = image_output.size[0] / image_output.size[1]
    src = []  # 源图像
    src_ratio = []  # 存储源图像的比例，后续要进行比对看和哪张图的原始比例最接近
    for i in range(len(img)):
        src.append(Image.open(os.path.join(images_folder, rescale_annos[i]['src_file_name'])))
        src_ratio.append(src[i].size[0] / src[i].size[1])
    eps = []
    for i in range(len(src_ratio)):
        eps.append(abs(ratio - src_ratio[i]))

    eps_index = sorted(range(len(eps)), key=lambda x: eps[x])[0]

    if eps[eps_index] < 0.5:
        w_ratio = src[eps_index].size[0] / image_output.size[0]
        h_ratio = src[eps_index].size[1] / image_output.size[1]
        image_output = image_output.resize((src[eps_index].size[0], src[eps_index].size[1]))
        for i in range(len(img)):
            rescale_annos[i]['bbox'] = [[v[0] * w_ratio, v[1] * h_ratio, v[2] * w_ratio, v[3] * h_ratio] for v in
                                        rescale_annos[i]['bbox']]
    else:
        w_ratio = src[eps_index].size[0] / image_output.size[0]
        new_h = round(w_ratio * image_output.size[1])
        image_output = image_output.resize((src[eps_index].size[0], new_h))
        for i in range(len(img)):
            rescale_annos[i]['bbox'] = [[v[0] * w_ratio, v[1] * w_ratio, v[2] * w_ratio, v[3] * w_ratio] for v in
                                        rescale_annos[i]['bbox']]

    for i in range(len(img)):
        if i == 0:
            pass
        else:
            l = len(rescale_annos[i - 1]['bbox'])
            rescale_annos[i]['hoi_sub'] = [v + l for v in rescale_annos[i]['hoi_sub']]
            rescale_annos[i]['hoi_obj'] = [v + l for v in rescale_annos[i]['hoi_obj']]
            rescale_annos[i]['bbox'] = rescale_annos[i - 1]['bbox'] + rescale_annos[i]['bbox']
            rescale_annos[i]['bbox_id'] = rescale_annos[i - 1]['bbox_id'] + rescale_annos[i]['bbox_id']
            rescale_annos[i]['hoi_sub'] = rescale_annos[i - 1]['hoi_sub'] + rescale_annos[i]['hoi_sub']
            rescale_annos[i]['hoi_obj'] = rescale_annos[i - 1]['hoi_obj'] + rescale_annos[i]['hoi_obj']
            rescale_annos[i]['verb_id'] = rescale_annos[i - 1]['verb_id'] + rescale_annos[i]['verb_id']
            rescale_annos[i]['hoi_id'] = rescale_annos[i - 1]['hoi_id'] + rescale_annos[i]['hoi_id']
            rescale_annos[i]['src_file_name'] = rescale_annos[i - 1]['src_file_name'] + rescale_annos[i][
                'src_file_name']

    return rescale_annos[len(rescale_images) - 1], image_output


def get_replace_image4_blending(random_index, annotations, images_folder, dataset_file):

    proc_anno = get_union_bbox(random_index, annotations, dataset_file)
    new_annos, new_imgaes = get_new_anno_and_image(len(random_index), proc_anno, images_folder)
    flip_annos, flip_images = random_flip_horizontal(len(random_index), new_annos, new_imgaes)
    rescale_annos, rescale_images = random_rescale(len(random_index), flip_annos, flip_images)
    anno, image = get_blending_images4(rescale_annos, rescale_images, images_folder)
    # plot_union_bbox(anno, image, 0)
    if dataset_file == 'vcoco':
        anno = convanno2vcoco(anno)
    elif dataset_file == 'hico':
        anno = convanno2hico(anno)
    else:
        raise Exception('not support dateset')
    return anno, image
#########单独测试blending有没有用###########################


# if __name__ == '__main__':
#     annotations_path = os.path.join('E:\\hico_20160224_det', 'annotations', 'trainval_hico.json')
#     images_folder = os.path.join('E:\\hico_20160224_det', 'images', 'train2015')
#
#     # annotations_path = os.path.join('E:\\v-coco', 'annotations', 'trainval_vcoco.json')
#     # images_folder = os.path.join('E:\\v-coco', 'images', 'train2014')
#
#     annotations = json.load(open(annotations_path))
#
#     sim_index = pickle.load(open('../datasets/sim_index_hico.pickle', 'rb'))
#
#
#     ##############################模拟测试运行状态################################
#     nohoi_index = []
#     for ind, img_anno in enumerate(annotations):
#         for hoi in img_anno['hoi_annotation']:
#             if hoi['subject_id'] >= len(img_anno['annotations']) or hoi['object_id'] >= len(img_anno['annotations']) or hoi['subject_id'] >= 100 or hoi['object_id'] >= 100:
#                 nohoi_index.append(ind)
#                 break
#
#     idx = np.random.randint(len(annotations))
#     while(idx in nohoi_index):
#         idx = np.random.randint(len(annotations))
#
#     sim_im_num = 3
#
#     random_index = [idx] + get_sim_index(sim_im_num, nohoi_index=nohoi_index, sim_index=sim_index[idx])
#     anno, img = get_replace_image(random_index, annotations, images_folder, dataset_file='hico')
#     ##############################模拟测试运行状态######################################
#
#     ####################保存随机状态复现报错的index###############################
#     # save_state = np.random.get_state()
#     # pickle.dump(save_state, open('random_state.pkl', 'wb'))
#
#     # save_state = pickle.load(open('random_state.pkl', 'rb'))
#     # np.random.set_state(save_state)
#     # np.random.seed(24)
#     ####################保存随机状态复现报错的index###############################