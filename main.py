import numpy as np
import cv2

def draw_rectangles(frame):
    rect_size = 100
    width, height, channels = frame.shape

    start_point = (int(height / 2 - rect_size / 2), int(width / 2 - rect_size / 2))
    end_point = (int(height / 2 + rect_size / 2), int(width / 2 + rect_size / 2))

    start_point2 = (int(0), int(0))
    end_point2 = (int(100), int(100))

    start_point3 = (int(370), int(370))
    end_point3 = (int(470), int(470))

    color = (0, 255, 0)
    thickness = 2

    rect = cv2.rectangle(frame, start_point, end_point, color, thickness)
    rect2 = cv2.rectangle(frame, start_point2, end_point2, color, thickness)
    rect3 = cv2.rectangle(frame, start_point3, end_point3, color, thickness)

    rectangle = []
    rectangle2 = []
    rectangle3 = []


    rectangle.append([start_point, end_point])
    rectangle2.append([start_point2, end_point2])
    rectangle3.append([start_point3, end_point3])

    return frame, rect, rect2, rect3, rectangle, rectangle2, rectangle3


def check_rectangle(frame, rectangle):
    h_sensivity = 20
    s_h = 255
    v_h = 255
    s_l = 50
    v_l = 50
    rect_size = 100
    thickness = 2
    color = (0, 255, 0)

    start_point = rectangle[0][0]
    end_point = rectangle[0][1]

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green_upper = np.array([60 + h_sensivity, s_h, v_h])
    green_lower = np.array([60 - h_sensivity, s_l, v_l])
    mask_frame = hsv_frame[start_point[1]:end_point[1] + 1, start_point[0]:end_point[0] + 1]
    mask_green = cv2.inRange(mask_frame, green_lower, green_upper)

    red_upper = np.array([5 + h_sensivity, s_h, v_h])
    red_lower = np.array([5 - h_sensivity, s_l, v_l])
    mask_red = cv2.inRange(mask_frame, red_lower, red_upper)

    orange_upper = np.array([20 + h_sensivity, s_h, v_h])
    orange_lower = np.array([20 - h_sensivity, s_l, v_l])
    mask_orange = cv2.inRange(mask_frame, orange_lower, orange_upper)

    yellow_upper = np.array([30 + h_sensivity, s_h, v_h])
    yellow_lower = np.array([30 - h_sensivity, s_l, v_l])
    mask_yellow = cv2.inRange(mask_frame, yellow_lower, yellow_upper)

    light_blue_upper = np.array([89 + h_sensivity, s_h, v_h])
    light_blue_lower = np.array([89 - h_sensivity, s_l, v_l])
    mask_light_blue = cv2.inRange(mask_frame, light_blue_lower, light_blue_upper)

    pink_upper = np.array([150 + h_sensivity, s_h, v_h])
    pink_lower = np.array([150 - h_sensivity, s_l, v_l])
    mask_pink = cv2.inRange(mask_frame, pink_lower, pink_upper)

    dark_blue_upper = np.array([120 + h_sensivity, s_h, v_h])
    dark_blue_lower = np.array([120 - h_sensivity, s_l, v_l])
    mask_dark_blue = cv2.inRange(mask_frame, dark_blue_lower, dark_blue_upper)

    green_rate = np.count_nonzero(mask_green) / (rect_size * rect_size)
    red_rate = np.count_nonzero(mask_red) / (rect_size * rect_size)
    orange_rate = np.count_nonzero(mask_orange) / (rect_size * rect_size)
    yellow_rate = np.count_nonzero(mask_yellow) / (rect_size * rect_size)
    light_blue_rate = np.count_nonzero(mask_light_blue) / (rect_size * rect_size)
    pink_rate = np.count_nonzero(mask_pink) / (rect_size * rect_size)
    dark_blue_rate = np.count_nonzero(mask_dark_blue) / (rect_size * rect_size)

    org = end_point
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.7

    if green_rate > 0.9:
        text = cv2.putText(frame, ' green ', org, font, fontScale, color, thickness, cv2.LINE_AA)
    elif red_rate > 0.9:
        text = cv2.putText(frame, ' red ', org, font, fontScale, color, thickness, cv2.LINE_AA)
    elif orange_rate > 0.9:
        text = cv2.putText(frame, ' orange ', org, font, fontScale, color, thickness, cv2.LINE_AA)
    elif yellow_rate > 0.9:
        text = cv2.putText(frame, ' yellow ', org, font, fontScale, color, thickness, cv2.LINE_AA)
    elif light_blue_rate > 0.9:
        text = cv2.putText(frame, ' light blue ', org, font, fontScale, color, thickness, cv2.LINE_AA)
    elif pink_rate > 0.9:
        text = cv2.putText(frame, ' pink ', org, font, fontScale, color, thickness, cv2.LINE_AA)
    elif dark_blue_rate > 0.9:
        text = cv2.putText(frame, ' dark blue ', org, font, fontScale, color, thickness, cv2.LINE_AA)
    else:
        text = cv2.putText(frame, ' not defined ', org, font, fontScale, color, thickness, cv2.LINE_AA)

    av_hue = np.average(mask_frame[:, :, 0])
    av_sat = np.average(mask_frame[:, :, 1])
    av_val = np.average(mask_frame[:, :, 2])
    average = [int(av_hue), int(av_sat), int(av_val)]

    text = cv2.putText(frame, str(average) + " " + str(green_rate), (10, 50), font, fontScale, color, thickness,
                       cv2.LINE_AA)
    frame = text
    return frame


def main():
    print('Press Esc to Quit the Application\n')

    cap = cv2.VideoCapture(-1)

    while cap.isOpened():
        ret, frame = cap.read() #1 - true 2 - result

        frame = cv2.flip(frame, 180) 

        frame, rect, rect2, rect3, rectangle, rectangle2, rectangle3 = draw_rectangles(frame)

        cv2.imshow('Login window', frame)

        x = cv2.waitKey(1) & 0xFF

        if x == 32: #space
            modified_frame = check_rectangle(frame, rectangle)
            cv2.imshow('Login window', modified_frame)
        elif x == 52: #4
            modified_frame = check_rectangle(frame, rectangle2)
            cv2.imshow('Login window', modified_frame)
        elif x == 113: #q
            modified_frame = check_rectangle(frame, rectangle3)
            cv2.imshow('Login window', modified_frame)


        # exit if "Esc" is pressed
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # Word Esc
            print('Good Bye!')
            break

    cap.release()
    cv2.destroyAllWindows()


main()
