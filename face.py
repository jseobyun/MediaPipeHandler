import os
import cv2
import numpy as np
import mediapipe as mp

class FaceDetector():
    def __init__(self, max_num_faces=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):

        self.joint_names = [
        ]
        self.skeleton=[
        ]
        self.num_joints = len(self.joint_names)
        self.num_skeleton = len(self.skeleton)

        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.max_num_faces = max_num_faces
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.contours = list(self.mp_face_mesh.FACEMESH_CONTOURS)[:1]


    def decode_results(self, img, results):
        img_h, img_w, _ = np.shape(img)
        face_lmks = results.multi_face_landmarks
        faces = []
        for lmk in face_lmks:
            xyz = np.zeros([self.num_joints, 3])
            for idx in range(self.num_joints):
                xyz[idx, 0] = lmk.landmark[idx].x * img_w
                xyz[idx, 1] = lmk.landmark[idx].y * img_h
                xyz[idx, 2] = lmk.landmark[idx].z
            faces.append(xyz)

        output_dict = {
            'face' : faces,
        }
        return output_dict

    def detect_on_image(self, cvimg):
        if isinstance(cvimg, str):
            cvimg = cv2.imread(cvimg)

        with self.mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=self.max_num_faces, min_detection_confidence=self.min_detection_confidence) as face_mesh:
            img = cvimg
            results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if not results.multi_face_landmarks:
                return None

            img_h, img_w, _ = np.shape(img)
            for face_landmarks in results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    image=img,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style())
                self.mp_drawing.draw_landmarks(
                    image=img,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style())
                self.mp_drawing.draw_landmarks(
                    image=img,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_iris_connections_style())

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

    def detect_on_video(self, video):
        if isinstance(video, str):
            video = cv2.VideoCapture(video)
        with self.mp_face_mesh.FaceMesh(max_num_faces=self.max_num_faces, refine_landmarks=True, min_detection_confidence=self.min_detection_confidence) as face_mesh:
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
                results = face_mesh.process(frame)

                # Draw the hand annotations on the image.
                frame.flags.writeable = True
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        # self.mp_drawing.draw_landmarks(
                        #     image=frame,
                        #     landmark_list=face_landmarks,
                        #     connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                        #     landmark_drawing_spec=None,
                        #     connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style())
                        self.mp_drawing.draw_landmarks(
                            image=frame,
                            landmark_list=face_landmarks,
                            connections=self.contours, #self.mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style())
                        # self.mp_drawing.draw_landmarks(
                        #     image=frame,
                        #     landmark_list=face_landmarks,
                        #     connections=self.mp_face_mesh.FACEMESH_IRISES,
                        #     landmark_drawing_spec=None,
                        #     connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_iris_connections_style())
                    output = self.decode_results(frame, results)
                else:
                    output = {
                        'face': None,
                    }
                # Flip the image horizontally for a selfie-view display.
                outputs.append(output)
                cv2.imshow('MediaPipe Face mesh', frame)
                if cv2.waitKey(5) & 0xFF == 27:
                    break

            video.release()
        return outputs

if __name__ =='__main__':
    HD = FaceDetector()
    #output = HD.detect_on_image('/home/jseob/Desktop/yjs/images/hand.jpg')
    outputs = HD.detect_on_video('/home/jseob/Desktop/yjs/images/body.mp4')


