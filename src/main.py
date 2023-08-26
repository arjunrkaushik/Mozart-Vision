import argparse
import math
import os

import cv2
import numpy as np
from keras.models import load_model

from note_classifier import train_classifier
from notation_tab import NotationTablature
from note import Note
from region import *
from staff_line import StaffLine

STAFF_LINE_THRESHOLD = 2
SEGMENTED_DIR = '../segmented'
ORG_SEGMENTED_DIR = '../org_segmented'
PITCH_ESTIMATION_DIR = '../pitch_estimation'
LINE_REMOVED_DIR = '../lin_removed'

staff_line_positions = []

input_file = "input.png"


def segment_image(for_classification=True, calculate_staff_line_pos=False):
    img = cv2.imread(input_file)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    gray_blur = cv2.blur(gray_img, (5, 5))

    edges = cv2.Canny(gray_blur, 80, 120)
    cdst = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)

    # lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    #
    # if lines is not None:
    #     for i in range(0, len(lines)):
    #         rho = lines[i][0][0]
    #         theta = lines[i][0][1]
    #         a = math.cos(theta)
    #         b = math.sin(theta)
    #         x0 = a * rho
    #         y0 = b * rho
    #         pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
    #         pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
    #         cv2.line(cdst, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)
    #
    # linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, None, 50, 10)
    #
    # if linesP is not None:
    #     for i in range(0, len(linesP)):
    #         l = linesP[i][0]
    #         cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)
    #
    # cv2.imshow("Source", img)
    # cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
    # cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)
    # cv2.waitKey(0)
    # Remove horizontal
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    for c in cnts:
        cv2.drawContours(img, [c], -1, (255, 255, 255), 2)
        min_x = math.inf
        max_x = -math.inf
        min_y = math.inf
        max_y = -math.inf
        for i in range(len(c)):
            min_x = min(c[i][0][0], min_x)
            max_x = max(c[i][0][0], max_x)
            min_y = min(c[i][0][1], min_y)
            max_y = max(c[i][0][1], max_y)
        if calculate_staff_line_pos:
            staff_line_positions.append(StaffLine(min_x, min_y, max_x, max_y))

    # Repair image
    repair_kernel_shape = cv2.MORPH_RECT if for_classification else cv2.MORPH_ELLIPSE
    repair_kernel_size = (1, 6) if for_classification else (10, 10)
    repair_kernel = cv2.getStructuringElement(repair_kernel_shape, repair_kernel_size)
    result = 255 - cv2.morphologyEx(255 - img, cv2.MORPH_CLOSE, repair_kernel, iterations=1)

    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    cropped_imgs = []
    bb_res = result.copy()
    for cntr in contours:
        x, y, w, h = cv2.boundingRect(cntr)
        cv2.rectangle(bb_res, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cropped_imgs.append(result[y:y + h, x:x + w])

    main_img = cropped_imgs[0]

    for cimg in cropped_imgs:
        if main_img.shape[0] * main_img.shape[1] < cimg.shape[0] * cimg.shape[1]:
            main_img = cimg

    # cv2.imshow("main_bb_img", main_img)
    # cv2.waitKey(0)

    gray_main_img = cv2.cvtColor(main_img, cv2.COLOR_BGR2GRAY)
    main_img = cv2.threshold(gray_main_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # cv2.imshow("thresh_main", main_img)
    # cv2.waitKey(0)

    fully_white_columns = []

    for j in range(main_img.shape[1]):
        is_column_blank = True
        for i in range(main_img.shape[0]):
            is_column_blank = is_column_blank and main_img[i][j] == 255

        if is_column_blank:
            fully_white_columns.append(j)

    crop_points = []

    l = 0
    r = l
    BLANK_SPACE_WIDTH_THRESHOLD = 10
    while l < len(fully_white_columns):
        while r + 1 < len(fully_white_columns) and fully_white_columns[r] + 1 == fully_white_columns[r + 1]:
            r += 1
        if r - l > BLANK_SPACE_WIDTH_THRESHOLD:
            crop_points.append(fully_white_columns[(l + r) // 2])
        l = r + 1
        r = l

    crop_points.insert(0, 0)
    crop_points.append(main_img.shape[1])

    segmented_images = []
    for cpi in range(len(crop_points) - 1):
        segmented_images.append(main_img[:, crop_points[cpi]:crop_points[cpi + 1]])

    # org_img =
    # org_segmented_images = []
    # for cpi in range(len(crop_points) - 1):
    #     org_segmented_images.append(org_img[:, crop_points[cpi]:crop_points[cpi + 1]])

    DIR_TYPE = SEGMENTED_DIR if for_classification else PITCH_ESTIMATION_DIR
    if not os.path.exists(DIR_TYPE):
        os.mkdir(DIR_TYPE)
    for seg_i in range(len(segmented_images)):
        cv2.imwrite(DIR_TYPE + '/' + str(seg_i + 1) + '.png', np.array(segmented_images[seg_i]))

    # if not os.path.exists(ORG_SEGMENTED_DIR):
    #     os.mkdir(ORG_SEGMENTED_DIR)
    # for seg_i in range(len(org_segmented_images)):
    #     cv2.imwrite(ORG_SEGMENTED_DIR + '/' + str(seg_i + 1) + '.png', np.array(org_segmented_images[seg_i]))


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_classifier", type=int, default=0)

    args = parser.parse_args()

    should_train_classifier = args.train_classifier == 1

    if should_train_classifier:
        train_classifier()

    segment_image(True, True)
    segment_image(False)
    # Morph kernel for filling half notes for pitch detection

    note_list = []

    for file in os.listdir(PITCH_ESTIMATION_DIR):
        image = cv2.imread(PITCH_ESTIMATION_DIR + "/" + file)

        kernel = np.ones((5, 5), np.uint8)

        img_dilation = cv2.dilate(image, kernel, iterations=1)

        gray_img = cv2.cvtColor(img_dilation, cv2.COLOR_BGR2GRAY)

        corner_img = cv2.cornerHarris(gray_img, 2, 3, 0.04)
        dst = cv2.dilate(corner_img, None)

        dst_corner_res = dst > 0.01 * dst.max()

        corner_coordinates = []
        for i in range(dst_corner_res.shape[0]):
            for j in range(dst_corner_res.shape[1]):
                if dst_corner_res[i][j]:
                    corner_coordinates.append((j, i))
        # image[dst_corner_res] = [0, 0, 255]

        corner_coordinates_arr = np.array(corner_coordinates)

        if len(corner_coordinates) > 10:
            sum_x = np.sum(corner_coordinates_arr[:, 0])
            sum_y = np.sum(corner_coordinates_arr[:, 1])
            centroid = sum_x // len(corner_coordinates), sum_y // len(corner_coordinates)
            note_list.append(Note(int(file[:-4]), file, image, centroid))
            # Blue color in BGR
            # color = (255, 0, 0)
            #
            # # Line thickness of 2 px
            # thickness = -1
            #
            # # Using cv2.circle() method
            # # Draw a circle with blue line borders of thickness of 2 px
            centroid_img = cv2.circle(image, centroid, 4, (255, 0, 0), -1)
            #
            # # Displaying the image
            # cv2.imshow("circle", image)
            #
            # cv2.imshow("centroid", centroid_img)
            # cv2.waitKey(0)
    # Pitch Estimation

    avg_distance_between_lines = 0

    total_dist = 0
    for pos_i in range(len(staff_line_positions) - 1):
        total_dist += abs(staff_line_positions[pos_i].y1 - staff_line_positions[pos_i + 1].y1)

    avg_distance_between_lines = total_dist / len(staff_line_positions) - 1

    pitch_mapping = {}
    SPACE_THRESHOLD = 0.75
    LINE_THRESHOLD = 1 - SPACE_THRESHOLD
    space_region_distance = SPACE_THRESHOLD * avg_distance_between_lines
    line_region_distance = LINE_THRESHOLD * avg_distance_between_lines

    staff_line_positions.sort(key=lambda x: x.y1)

    line_region_list = []
    for i in range(len(staff_line_positions)):
        y = staff_line_positions[i].y1
        line_region_list.append(Region(
            "l" + str(i + 1),
            "line",
            y - (line_region_distance / 2),
            y + (line_region_distance / 2)
        ))

    space_region_list = []
    for i in range(len(line_region_list) - 1):
        upper = line_region_list[i].lower_bound + 1
        lower = line_region_list[i + 1].upper_bound - 1
        space_region_list.append(Region(
            "s" + str(i + 2),  # 2 cuz we are adding s1 and s_last in the end
            "space",
            upper,
            lower
        ))

    region_list = []

    for i in range(len(line_region_list)):
        region_list.append(line_region_list[i])
        if i < len(space_region_list):
            region_list.append(space_region_list[i])

    # Add buffer spaces
    first_line_region = region_list[0]
    region_list.insert(
        0,
        Region(
            "s1",
            "space",
            first_line_region.upper_bound - space_region_distance,
            first_line_region.upper_bound - 1
        )
    )
    last_line_region = region_list[len(region_list) - 1]
    region_list.append(
        Region(
            "s" + str(len(region_list)),
            "space",
            last_line_region.lower_bound + 1,
            last_line_region.lower_bound + space_region_distance
        )
    )

    # Visualize
    visual_img = cv2.imread(input_file)
    for rgn in region_list:
        color1 = (0, 255, 0)
        color2 = (0, 0, 255)
        # Line thickness of -1 px
        # Thickness of -1 will fill the entire shape
        thickness = -1

        if rgn.region_type == "line":
            color = color2
        else:
            color = color1
        visual_img = cv2.rectangle(
            visual_img,
            (50, int(rgn.upper_bound)),
            (visual_img.shape[1] - 50, int(rgn.lower_bound)),
            color,
            thickness
        )

    # cv2.imshow("Regions", visual_img)
    # cv2.waitKey(0)

    note_list.sort(key=lambda x: x.segment_id)

    # Popping first two notes since they are clefs and time signature. Works for now
    note_list.pop(0)
    note_list.pop(0)

    notation_results = []
    for note in note_list:
        min_dist = math.inf
        min_i = 0
        for i in range(len(region_list)):
            region = region_list[i]
            region_mid_y = region.upper_bound + abs(region.upper_bound - region.lower_bound) // 2
            note_centroid_y = note.centroid[1]
            dist_to_region = abs(region_mid_y - note_centroid_y)
            if dist_to_region < min_dist:
                min_dist = dist_to_region
                min_i = i

        notation_results.append(NotationTablature(note, region_list[min_i].name))

    # Classifier
    model_path = "../model/note_classifier_model.h5"

    try:
        model = load_model(model_path)
    except:
        raise Exception("Train the model first by running the main script with train_classifier parameter as 1")

    def get_note_type(pred_class: int):
        pred_class_map = {0: "Eight", 1: "Half", 2: "Quarter", 3: "Sixteenth", 4: "Whole"}
        if pred_class < 0 or pred_class > 4:
            raise Exception("Prediction index out of bounds")
        return pred_class_map[pred_class]

    for result in notation_results:
        pitch_estimated_file = result.note.filename

        img = cv2.imread(SEGMENTED_DIR + "/" + pitch_estimated_file, 0)

        # while img.shape[0] < 256 or img.shape[1] < 256:
        #     if img.shape[0] < 256:
        #         img = np.vstack([np.full([1, img.shape[1]], 256), img])
        #         img = np.vstack([img, np.full([1, img.shape[1]], 256)])
        #     else:
        #         img = np.hstack([np.full([img.shape[0], 1], 256), img])
        #         img = np.hstack([img, np.full([img.shape[0], 1], 256)])

        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)

        pred = model.predict(np.expand_dims(img / 255, 0))

        note_type = get_note_type(pred.argmax(axis=1)[0])

        result.set_note_type(note_type)

    # Output
    from notations import notation_map

    print("\n-----The staff/ music sheet notation translates to the following tablature:-----\n")
    for out in notation_results:
        result = notation_map[out.staff_region]
        print("image: " + out.note.filename)
        print("Pitch: " + result.pitch)
        print("Fret: " + str(result.fret))
        print("String: " + str(result.string))
        print("Note Type: " + out.note_type)
        print("\n")

    # Clean up
    # shutil.rmtree(SEGMENTED_DIR)
    # shutil.rmtree(PITCH_ESTIMATION_DIR)


if __name__ == "__main__":
    run()
