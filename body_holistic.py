import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm

class WholeBodyDetector():
    def __init__(self, model_complexity=2, enable_segmentation=True, min_detection_confidence=0.5, min_tracking_confidence=0.5):

        self.joint_names = [
            'wrist',
        ]
        self.skeleton=[

        ]
        self.num_joints = len(self.joint_names)
        self.num_skeleton = len(self.skeleton)

        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.model_complexity = model_complexity
        self.enable_segmentation = enable_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.counter = 0

    def decode_results(self, img, results):
        img_h, img_w, _ = np.shape(img)
        face = results.face_landmarks
        lhand = results.left_hand_landmarks
        rhand = results.right_hand_landmarks
        body = results.pose_landmarks
        body3d = results.pose_world_landmarks
        mask = results.segmentation_mask
        if face is not None:
            face_np = np.zeros([len(face.landmark), 3])
            for fidx in range(len(face.landmark)):
                face_np[fidx, 0] = face.landmark[fidx].x * img_w
                face_np[fidx, 1] = face.landmark[fidx].y * img_h
                face_np[fidx, 2] = face.landmark[fidx].z * img_w # The magnitude of z uses roughly the same scale as x.
        else:
            face_np = None

        if lhand is not None:
            lhand_np = np.zeros([len(lhand.landmark), 3])
            for hidx in range(len(lhand.landmark)):
                lhand_np[hidx, 0] = lhand.landmark[hidx].x * img_w
                lhand_np[hidx, 1] = lhand.landmark[hidx].y * img_h
                lhand_np[hidx, 2] = lhand.landmark[hidx].z * img_w  # The magnitude of z uses roughly the same scale as x.
        else:
            lhand_np = None

        if rhand is not None:
            rhand_np = np.zeros([len(rhand.landmark), 3])
            for hidx in range(len(rhand.landmark)):
                rhand_np[hidx, 0] = rhand.landmark[hidx].x * img_w
                rhand_np[hidx, 1] = rhand.landmark[hidx].y * img_h
                rhand_np[hidx, 2] = rhand.landmark[hidx].z * img_w # The magnitude of z uses roughly the same scale as x.
        else:
            rhand_np = None

        if body is not None:
            body_np = np.zeros([len(body.landmark), 4])
            for bidx in range(len(body.landmark)):
                body_np[bidx, 0] = body.landmark[bidx].x * img_w
                body_np[bidx, 1] = body.landmark[bidx].y * img_h
                body_np[bidx, 2] = body.landmark[bidx].z * img_w  # The magnitude of z uses roughly the same scale as x.
                body_np[bidx, 3] = body.landmark[bidx].visibility
        else:
            body_np = None

        if body3d is not None:
            body3d_np = np.zeros([len(body.landmark), 4])
            for bidx in range(len(body3d.landmark)):
                body3d_np[bidx, 0] = body3d.landmark[bidx].x
                body3d_np[bidx, 1] = body3d.landmark[bidx].y
                body3d_np[bidx, 2] = body3d.landmark[bidx].z # The magnitude of z uses roughly the same scale as x.
                body3d_np[bidx, 3] = body3d.landmark[bidx].visibility
        else:
            body3d_np = None

        if mask is not None:
            mask_np = results.segmentation_mask
        else:
            mask_np = None
        output_dict = {
            'face': face_np,
            'lhand': lhand_np,
            'rhand': rhand_np,
            'body': body_np,
            'body3d': body3d_np,
            'mask' : mask_np,
        }
        return output_dict

    def detect_on_image(self, cvimg):
        if isinstance(cvimg, str):
            cvimg = cv2.imread(cvimg)

        with self.mp_holistic.Holistic(
                static_image_mode=True,
                enable_segmentation=self.enable_segmentation,
                min_detection_confidence=self.min_detection_confidence) as holistic:
            frame = cvimg
            results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img_h, img_w, _ = np.shape(frame)
            self.mp_drawing.draw_landmarks(
                frame,
                results.face_landmarks,
                self.mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style())
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
            self.mp_drawing.draw_landmarks(
                frame, results.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style()
            )
            self.mp_drawing.draw_landmarks(
                frame, results.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style()
            )
            output = self.decode_results(frame, results)
            return output

    def detect_on_dir(self, dir):
        file_names = os.listdir(dir)
        file_names = [file_name for file_name in file_names if file_name.endswith('.png') or file_name.endswith('.jpg')]
        file_paths = [os.path.join(dir, file_name) for file_name in file_names]
        outputs = []
        for file_path in tqdm(file_paths):
            output = self.detect_on_image(file_path)
            outputs.append(output)
        return outputs

    def detect_on_video(self, video):
        if isinstance(video, str):
            video = cv2.VideoCapture(video)
        with self.mp_holistic.Holistic(
                enable_segmentation=self.enable_segmentation,
                min_detection_confidence=self.min_detection_confidence) as holistic:
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
                results = holistic.process(frame)

                # Draw the hand annotations on the image.
                frame.flags.writeable = True
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                self.mp_drawing.draw_landmarks(
                    frame,
                    results.face_landmarks,
                    self.mp_holistic.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style())
                self.mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    self.mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
                self.mp_drawing.draw_landmarks(
                    frame, results.left_hand_landmarks,
                    self.mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style()
                )
                self.mp_drawing.draw_landmarks(
                    frame, results.right_hand_landmarks,
                    self.mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style()
                )
                output = self.decode_results(frame, results)
                outputs.append(output)
                # Flip the image horizontally for a selfie-view display.
                cv2.imshow('MediaPipe Holistic', frame)
                if cv2.waitKey(5) & 0xFF == 27:
                    break

            video.release()
        return outputs

if __name__ =='__main__':
    BD = WholeBodyDetector()
    #output = BD.detect_on_image('/home/jseob/Desktop/yjs/images/person.png')

    outputs = BD.detect_on_video('/home/jseob/Desktop/yjs/images/body.mp4')
    print("")
    #output = BD.detect_on_dir('/media/jseob/SSD_yjs/jseob/MSCOCO/images/val2017')


