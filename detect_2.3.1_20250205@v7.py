import argparse
import os
import glob
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

names_list = ['B1','B2','B3','B4','B5','BO','BS','R1','R2','R3','R4','R5','RO','RS']

last_time = time.time()
def detect(opt):
    global last_time
    source, weights, view_img, save_txt, save_frames, imgsz, save_txt_tidl, kpt_label, ourteam, imgx, imgy = opt.source, opt.weights, opt.view_img, \
        opt.save_txt, opt.save_frames, opt.img_size, \
        opt.save_txt_tidl, opt.kpt_label, opt.ourteam, opt.imgx, opt.imgy
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    print("正在开始检测")
    # cv2.namedWindow('detect_v1', cv2.WINDOW_NORMAL)
    print(f"source: {source}, weights: {weights}, view_img: {view_img}, save_txt: {save_txt}, save_frames: {save_frames}, imgsz: {imgsz}, save_txt_tidl: {save_txt_tidl}, kpt_label: {kpt_label}, save_img: {save_img}")
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = increment_path(
        Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
    (save_dir / 'labels' if (save_txt or save_txt_tidl)
     else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    (save_dir / 'frames').mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    # half precision only supported on CUDA
    half = device.type != 'cpu' and not save_txt_tidl

    total_frames = 0  # for save frames

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    if isinstance(imgsz, (list, tuple)):
        assert len(imgsz) == 2
        "height and width of image has to be specified"
        imgsz[0] = check_img_size(imgsz[0], s=stride)
        imgsz[1] = check_img_size(imgsz[1], s=stride)
    else:
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(
        model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load(
            'weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(
            next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        # pred = model(img, augment=opt.augment)[0]
        pred = model(img)[0]

        # Apply NMS
        # print(pred[:,1], "e")
        if kpt_label:
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                                       agnostic=opt.agnostic_nms,
                                       kpt_label=kpt_label, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'])
        else:
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                                       agnostic=opt.agnostic_nms, kpt_label=kpt_label, nc=model.yaml['nc'])
        t2 = time_synchronized()
        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(
                ), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + \
                ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            # normalization gain whwh
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            if len(det):
                save_img = True
                # Rescale boxes from img_size to im0 size
                output_dir = str(save_dir / 'frames')
                if save_frames:
                    cv2.imwrite(str(save_dir / 'frames' /
                                f'{total_frames}.jpg'), im0)
                    total_frames = total_frames + 1

                scale_coords(img.shape[2:], det[:, :4],
                             im0.shape, kpt_label=False)
                scale_coords(img.shape[2:], det[:, 6:],
                             im0.shape, kpt_label=kpt_label, step=3)

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    # add to string
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                # Write results
                for det_index, (*xyxy, conf, cls) in enumerate(det[:, :6]):
                    kpts = det[det_index, 6:]

                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    # label format
                    line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)
                    # Normalize keypoints
                    im_width = im0.shape[1]
                    im_height = im0.shape[0]
                    kpts = det[det_index, 6:]
                    # print(f"\n原始关键点 Original keypoints: {kpts}")  # 打印未归一化的关键点
                    norm_kpts = [kpts[i] / im_width if i % 3 == 0 else kpts[i] / im_height if i % 3 == 1 else kpts[i]
                                    for i in range(len(kpts))]

                    line += tuple(norm_kpts)

                if save_img:
                    result_list = det.tolist()
                    # print(f"目标个数{len(result_list)}")
                    # print(f"正在获取图片 Saving image: {save_path}\nNormalized keypoints: {list(norm_kpts)}")
                    fps = 1
                    if not len(norm_kpts) % 12:
                        # print(f"检测到 {len(norm_kpts)} Key points number is correct")
                        # Get all image paths
                        image_paths = glob.glob(os.path.join(source, "*"))
                        image_path = image_paths[total_frames - 1]
                        if image_path:
                            image = im0s.copy()
                            if not os.path.exists(f"{output_dir}/_results_/"):
                                os.makedirs(f"{output_dir}/_results_/")
                            output_path = f"{output_dir}/_results_/{total_frames}.jpg"
                            # print(f"保存的 _results_ : {output_path}")
                            # print(f"{result_list}")

                            max_area = {}
                            for _ in result_list:
                                result = _[:4]
                                x_min, y_min, x_max, y_max = map(int, result)
                                area = (x_max - x_min) * (y_max - y_min)
                                average_point = ((x_max + x_min) / 2, (y_max + y_min) / 2)


                                # 绘制矩形框
                                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                                # print(f"绘制矩形框 Rectangle: {x_min, y_min, x_max, y_max}")

                                # 在框上方绘制标
                                label = names_list[int(torch.tensor(cls).view(1).tolist()[0])]
                                # print(f"{cls} label={label}")

                                if ourteam == 'blue':
                                    bg = 41 # 红
                                    if cls > 6.5:
                                        max_area[area] = [label, average_point, area]
                                    # if len(max_area) == 0 and cls > 6.5:
                                    #     max_area[area] = [label, average_point, area]
                                    # elif area > max(max_area) and cls > 6.5:
                                    #     max_area[area] = [label, average_point, area]
                                        # cls 0 ~ 6: Blue, cls 7~13: Red
                                        # names_list = ['B1','B2','B3','B4','B5','BO','BS','R1','R2','R3','R4','R5','RO','RS']
                                else:
                                    bg = 44 # 蓝

                                cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                                keypoints = [(_[i], _[i+1], _[i+2]) for i in range(6, len(_), 3)]
                                # print(keypoints)
                                for (kx, ky, kc) in keypoints:
                                    # print(f"绘制关键点 Key point: {kx, ky, kc}")
                                    if kc > 0:
                                        cv2.circle(image, (int(kx), int(ky)), 5, (0, 0, 255), -1)
                                # 绘制线段
                                # if len(keypoints) >= 4 and all(kc*x*y > 0 for x, y, kc in keypoints[:4]):
                                #     cv2.line(image, (int(keypoints[0][0]), int(keypoints[0][1])), (int(keypoints[2][0]), int(keypoints[2][1])), (0, 255, 0), 2)
                                #     cv2.line(image, (int(keypoints[1][0]), int(keypoints[1][1])), (int(keypoints[3][0]), int(keypoints[3][1])), (0, 255, 0), 2)
                                # if len(keypoints) >= 4:
                                #     cv2.line(image, (int(keypoints[0][0]), int(keypoints[0][1])), (int(keypoints[2][0]), int(keypoints[2][1])), (0, 255, 0), 2)
                                #     cv2.line(image, (int(keypoints[1][0]), int(keypoints[1][1])), (int(keypoints[3][0]), int(keypoints[3][1])), (0, 255, 0), 2)
                                #     cv2.line(image, (int(keypoints[0][0]), int(keypoints[0][1])), (int(keypoints[1][0]), int(keypoints[1][1])), (0, 255, 0), 2)
                                #     cv2.line(image, (int(keypoints[2][0]), int(keypoints[2][1])), (int(keypoints[3][0]), int(keypoints[3][1])), (0, 255, 0), 2)

                            # print(f"{max_area}")
                            # bg = 44 # 蓝
                            # bg = 41 # 红
                            if max_area:
                                # 获取最大的键
                                max_key = max(max_area.keys())
                                # 获取最大键对应的值
                                max_value = max_area[max_key]
                                # 提取average_point
                                average_point = max_value[1]
                            # if max_area:
                            #     for k, v in enumerate(max_area):
                            #         average_point = v[1]
                                print(f"\n\033[{bg};{38}m   {average_point}   \033[0m")
                                x_a, y_a = average_point
                                print(f"main|{x_a}|{y_a}")
                                # x_a, y_a = average_point
                                x, y = imgx, imgy
                                x = x / 2
                                y = y / 2
                                x_d = x_a - x
                                y_d = y_a - y
                                print(f"diff|{x_d}|{y_d}\n")

                            # 保存带有检测框和关键点的图片
                            # cv2.imwrite(output_path, image)
                            # 计算新的高度
                            height, width = image.shape[:2]
                            new_width = 960
                            scale_ratio = new_width / width
                            new_height = int(height * scale_ratio)
                            image = cv2.resize(image, (new_width, new_height))
                            current_time = time.time()
                            elapsed_time = current_time - last_time
                            elapsed_time_ms = (current_time - last_time) * 1000
                            # print(f"\nFPS: {1/elapsed_time:.2f} Inference time: {elapsed_time* 1000:.2f} ms 等待 {int(1000/fps - elapsed_time * 1000)}\n")
                            elapsed_time_str = f"FPS: {1000/elapsed_time_ms:.2f}  Inference I/O: {elapsed_time_ms:.2f} ms  Waiting {int(1000/fps - elapsed_time_ms)} ms"
                            image_copy = image.copy()
                            # cv2.putText(image_copy, elapsed_time_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (197, 122, 250), 2)
                            # cv2.imshow('detect_v1', image_copy)
                            height, width = image.shape[:2]
                            # cv2.resizeWindow('detect_v1', width, height)
                            # cv2.waitKey(max(1, int(1000/fps - elapsed_time * 1000)))
                            last_time = time.time()
                            # print(f"Output saved to {output_path}")

                    if save_txt:  # Write to file
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                        # kpts_list = (kpts.view(-1).tolist()
                        #              ) if kpts is not None else []
                        # # label format
                        # line = (
                        #     cls, *xywh, conf, *kpts_list) if opt.save_conf else (cls, *xywh, *kpts_list)
                        # with open(txt_path + '.txt', 'a') as f:
                        #     f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or opt.save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if opt.hide_labels else (
                            names[c] if opt.hide_conf else f'{names[c]} {conf:.2f}')
                        kpts = det[det_index, 6:]
                        if c == 0:
                            plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=1, kpts=kpts,
                                         steps=3)
                        if opt.save_crop:
                            save_one_box(
                                xyxy, im0s, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Print time (inference + NMS)
            # print(f'{s}Done. ({t2 - t1:.3f}s)')
            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                # print(f"正在保存图片 Saving image: {save_path}")
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_txt_tidl or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt or save_txt_tidl else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default="best_v1.3.pt", help='model.pt path(s)')
    parser.add_argument(
        '--source', type=str, default="test", help='source')
    parser.add_argument('--ourteam', type=str, default='blue')
    parser.add_argument('--imgx', type=int, default=1280)
    parser.add_argument('--imgy', type=int, default=768)
    parser.add_argument('--img-size', nargs='+', type=int,
                        default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.4, help='IOU threshold for NMS')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true',
                        help='display results')
    parser.add_argument('--save-txt', default=True, action='store_true',
                        help='save results to *.txt')
    parser.add_argument('--save-frames', default=True, action='store_true',
                        help='save detect frames to *.jpg')
    parser.add_argument('--save-txt-tidl', default=True, action='store_true',
                        help='save results to *.txt in tidl format')
    parser.add_argument('--save-bin', default=True, action='store_true',
                        help='save base n/w outputs in raw bin format')
    parser.add_argument('--save-conf', action='store_true',
                        help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', default=True, action='store_true',
                        help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true',
                        help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument(
        '--project', default='win_kpt/runs/detect', help='save to project/name')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3,
                        type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False,
                        action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False,
                        action='store_true', help='hide confidences')
    parser.add_argument('--kpt-label', default=True,
                        help='use keypoint labels for training')
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('tensorboard', 'pycocotools', 'thop'))
    detect(opt=opt)
