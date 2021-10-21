import os
import cv2
import numpy as np
import mediapipe as mp

class HandDetector():
    def __init__(self, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):

        self.joint_names = [
            'wrist',
            'thumb1', 'thumb2', 'thumb3', 'thumb4', # 1 2 3 4 denotes CMC MCP IP TIP
            'index1', 'index2', 'index3', 'index4', # 1 2 3 4 denotes MCP PIP DIP TIP
            'middle1', 'middle2', 'middle3', 'middle4',
            'ring1', 'ring2', 'ring3', 'ring4',
            'pinky1', 'pinky2', 'pinky3', 'pinky4',
        ]
        self.skeleton=[
            [0, 1], [1, 2], [2, 3], [3, 4],
            [0, 5], [5, 6], [6, 7], [7, 8],
            [0, 9], [9, 10], [10, 11], [11, 12],
            [0, 13], [13, 14], [14, 15], [15, 16],
            [0, 17], [17, 18], [18, 19], [19, 20],
        ]
        self.num_joints = len(self.joint_names)
        self.num_skeleton = len(self.skeleton)

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

    def decode_results(self, img, results):
        img_h, img_w, _ = np.shape(img)
        hand_lmks = results.multi_hand_landmarks
        hands = []
        for lmk in hand_lmks:
            xyz = np.zeros([self.num_joints, 3])
            for idx in range(self.num_joints):
                xyz[idx, 0] = lmk.landmark[idx].x * img_w
                xyz[idx, 1] = lmk.landmark[idx].y * img_h
                xyz[idx, 2] = lmk.landmark[idx].z
            hands.append(xyz)

        hand_labels = results.multi_handedness
        indice = []
        is_lefts = []

        for label in hand_labels:
            index = label.classification[0].index
            score = label.classification[0].score
            is_left = label.classification[0].label == 'Left'
            indice.append(index)
            is_lefts.append(is_left)
        output_dict = {
            'hands' : hands,
            'indice' : indice,
            'is_lefts' : is_lefts
        }
        return output_dict

    def detect_on_image(self, cvimg, flipped=True):
        if isinstance(cvimg, str):
            cvimg = cv2.imread(cvimg)

        with self.mp_hands.Hands(static_image_mode=True, max_num_hands=self.max_num_hands, min_detection_confidence=self.min_detection_confidence) as hands:
            if flipped:
                img = cv2.flip(cvimg, 1)
            else:
                img = cvimg
            results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if not results.multi_hand_landmarks:
                return None

            img_h, img_w, _ = np.shape(img)
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    img,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())

            return self.decode_results(img, results)

    def detect_on_dir(self, dir):
        file_names = os.listdir(dir)
        file_names = [file_name for file_name in file_names if file_name.endswith('.png') or file_name.endswith('.jpg')]
        file_paths = [os.path.join(dir, file_name) for file_name in file_names]
        outputs = []
        for file_path in file_paths:
            output = self.detect_on_image(file_path)
            outputs.append(output)
        return outputs

    def detect_on_video(self, video, flipped=True):
        if isinstance(video, str):
            video = cv2.VideoCapture(video)
        with self.mp_hands.Hands(max_num_hands=self.max_num_hands, min_detection_confidence=self.min_detection_confidence) as hands:
            if not video.isOpened():
                raise Exception(f"No {video} exists or file is not valid.")
            outputs = []
            while video.isOpened():
                success, frame = video.read()
                if not success:
                    print("Reading fail")
                    break
                frame.flags.writeable = False
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame)

                # Draw the hand annotations on the image.
                frame.flags.writeable = True
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style())
                    output = self.decode_results(frame, results)
                else:
                    output = {
                        'hands': None,
                        'indice': None,
                        'is_lefts': None,
                    }
                # Flip the image horizontally for a selfie-view display.
                cv2.imshow('MediaPipe Hands', cv2.flip(frame, 1))
                outputs.append(output)
                if cv2.waitKey(5) & 0xFF == 27:
                    break
            video.release()
        return outputs

if __name__ =='__main__':
    HD = HandDetector()
    output = HD.detect_on_image('/home/jseob/Desktop/yjs/images/hand.jpg')
    outputs = HD.detect_on_video('/home/jseob/Desktop/yjs/images/hands.mp4')


