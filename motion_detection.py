import multiprocessing
import os.path
import subprocess
from datetime import datetime
from itertools import repeat
from tkinter import Tk, filedialog

import cv2

base_dir = os.path.dirname(os.path.abspath(__file__))


def main(initial_directory):

    print('Select input directory')
    root = Tk()
    root.withdraw()
    directory = filedialog.askdirectory(initialdir=initial_directory)
    files_to_process = []
    for file in os.scandir(directory):
        if isinstance(file, os.DirEntry) and file.is_file() and not file.name.startswith('.'):
            files_to_process.append(file.path)
    if files_to_process:
        trim_points = get_trim_points(files_to_process[0])
        start = datetime.now()
        pool = multiprocessing.Pool(4)
        pool.starmap(process, zip(files_to_process, repeat(trim_points)))
        print(f'executed in {datetime.now() - start}')


def convert_millis(millis):
    seconds = (millis / 1000) % 60
    minutes = (millis / (1000 * 60)) % 60
    return "%d%02d" % (minutes, seconds)


def click_select(event, x, y, flags, data):
    image_container, points = data
    x = max(x, 0)
    x = min(x, image_container[0].shape[1])
    y = max(y, 0)
    y = min(y, image_container[0].shape[0])
    if event == cv2.EVENT_LBUTTONDOWN:
        points.clear()
        points.append((x, y))
    elif event == cv2.EVENT_LBUTTONUP:
        points.append((x, y))
        image = image_container[1].copy()
        cv2.rectangle(image, points[-2], points[-1], (0, 0, 255), 2)
        cv2.imshow('SELECT', image)
        image_container[0] = image


def show_mouse_select(image):
    image = [image, image]
    cv2.namedWindow('SELECT')

    points = []
    cv2.setMouseCallback('SELECT', click_select, (image, points))

    while True:
        cv2.imshow('SELECT', image[0])
        key = cv2.waitKey(1)
        if key == ord('\r'):
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            break  # press enter to exit

    # Output points and save image
    points_to_return = []
    if len(points) > 1:
        print('Points:')
        for i in range(0, len(points), 2):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]

            points_to_return.append((min(x1, x2), min(y1, y2)))
            points_to_return.append((max(x1, x2), max(y1, y2)))
            print(points_to_return)
    return points_to_return


def get_trim_points(file_name):
    if not os.path.isfile(file_name):
        return None
    video = cv2.VideoCapture(file_name)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    set_frame = round(frame_count / 2)
    video.set(cv2.CAP_PROP_POS_FRAMES, set_frame)
    grabbed, frame = video.read()
    video.release()
    if grabbed:
        frame = cv2.resize(frame, (1280, 720), cv2.INTER_AREA)
        return show_mouse_select(frame)


def get_start_iframe(iframes, start_position):
    start_iframe = iframes[0]
    for iframe in iframes:
        if start_position < iframe:
            break
        else:
            start_iframe = iframe
    return start_iframe


def prepare_output_dir(name):
    out_dir_name = f'{base_dir}/out/{name}'
    os.makedirs(out_dir_name, exist_ok=True)
    for to_delete in os.scandir(out_dir_name):
        if isinstance(to_delete, os.DirEntry) and to_delete.is_file():
            os.remove(to_delete)


def process(file_name, points):
    if not os.path.isfile(file_name):
        return None
    iframe_process = ffprobe_get_iframes_start(file_name)
    processes = []
    name = os.path.basename(file_name).rsplit('.', 1)[0]
    stat = os.stat(file_name)
    if hasattr(stat, 'st_birthtime'):
        creation_date = stat.st_birthtime
    else:
        creation_date = stat.st_ctime
    creation_date = datetime.fromtimestamp(creation_date).strftime('%Y%m%d')
    print(f'Processing {name}')

    # Prepare output dir
    out_dir_name = f'{base_dir}/out/{name}'
    os.makedirs(out_dir_name, exist_ok=True)
    for to_delete in os.scandir(out_dir_name):
        if isinstance(to_delete, os.DirEntry) and to_delete.is_file():
            os.remove(to_delete)

    iframes = None
    video = cv2.VideoCapture(file_name)
    background_0 = None
    background_1 = None
    output_start = None
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    end_frame_count = 0

    with open(f'{out_dir_name}/report.txt', 'w') as report:
        while video.isOpened():
            current_frame = video.get(cv2.CAP_PROP_POS_FRAMES)
            grabbed, frame = video.read()
            while not grabbed and current_frame < frame_count:
                current_frame = video.get(cv2.CAP_PROP_POS_FRAMES)
                grabbed, frame = video.read()
            if not grabbed:
                break

            # Convert format to enable GPU acceleration if available
            frame = cv2.UMat(frame)
            # resize the frame, convert it to grayscale, and blur it
            frame = cv2.resize(frame, (1280, 720), cv2.INTER_AREA)
            if len(points) >= 2:
                frame = cv2.UMat(frame, [points[0][1], points[1][1]], [points[0][0], points[1][0]])
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame = cv2.GaussianBlur(frame, (11, 11), 0)

            if background_1 is None:
                background_1 = frame
                continue
            if background_0 is None:
                background_0 = background_1
                background_1 = frame
                continue

            # compute the absolute difference between the current frame and first frame
            frame_delta = cv2.absdiff(background_0, frame)
            frame_delta = cv2.threshold(frame_delta, 15, 255, cv2.THRESH_BINARY)[1]

            # dilate the threshold image to fill in holes, then find contours on threshold image
            frame_delta = cv2.dilate(frame_delta, None, iterations=2)
            contours = cv2.findContours(frame_delta, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

            # Set background for next frame
            background_0 = background_1
            background_1 = frame

            # add current_ms to queue
            current_ms = video.get(cv2.CAP_PROP_POS_MSEC)

            if len(contours):
                # save to output
                if output_start is None:
                    output_start = current_ms
                    start_position = convert_millis(output_start)
                    out_file_name = f'{creation_date}-{start_position}.mp4'
                    out_file_path = f'{out_dir_name}/{out_file_name}'
                    report.write(f'{out_file_name} {start_position} {current_frame} ')
                end_frame_count = 0
            else:
                end_frame_count += 1
                if end_frame_count > fps * 2:
                    # remove to prevent overflow
                    if output_start is None:
                        end_frame_count -= 1
                    else:
                        # save the queue output file
                        end_frame_count = 0
                        if iframes is None:
                            iframes = ffprobe_get_iframes_finish(iframe_process)
                        start_time = get_start_iframe(iframes, output_start)
                        processes.append(ffmpeg_extract_clip(file_name, out_file_path, start_time, current_ms))
                        print(f'Start writing {name}/{out_file_name}')
                        output_start = None
                        finish_position = convert_millis(current_ms)
                        report.write(f'{finish_position} {current_frame}\n')

        if output_start is not None:
            video.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
            current_ms = video.get(cv2.CAP_PROP_POS_MSEC)
            if iframes is None:
                iframes = ffprobe_get_iframes_finish(iframe_process)
            start_time = get_start_iframe(iframes, output_start)
            processes.append(ffmpeg_extract_clip(file_name, out_file_path, start_time, current_ms))
            finish_position = convert_millis(current_ms)
            report.write(f'{finish_position}\n')
    cv2.destroyAllWindows()
    for p in processes:
        if p.poll() is None:
            print(f'Waiting for process to complete')
            p.wait()


# region ffmpeg
def ffprobe_get_iframes_start(original_file):
    args = ['ffprobe', '-hide_banner', '-loglevel', 'fatal', '-select_streams', 'v', '-show_entries',
            'packet=pts_time,flags', '-of', 'csv=print_section=0', original_file]
    return subprocess.Popen(args, stdout=subprocess.PIPE)


def ffprobe_get_iframes_finish(process_out):
    iframes = []
    std_out, std_err = process_out.communicate()
    std_out = std_out.decode()
    for process_line in std_out.split('\n'):
        items = process_line.split(',')
        if len(items) >= 2 and 'K' in items[1]:
            value = round(float(items[0]) * 1000, 3)
            iframes.append(value)
    return iframes


def ffmpeg_extract_clip(original_file, output_file, start_ms, stop_ms):
    return subprocess.Popen(
        ['ffmpeg', '-hide_banner', '-loglevel', 'fatal', '-y', '-i', original_file, '-ss', f'{start_ms}ms', '-to',
         f'{stop_ms}ms', '-c', 'copy', output_file])

# endregion
