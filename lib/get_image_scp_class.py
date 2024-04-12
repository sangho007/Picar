import cv2
import numpy as np
import paramiko
import scp
import math
from sklearn.cluster import KMeans
from datetime import datetime
import constant
import ast

class CameraCalibrator:
    def __init__(self):
        self.dist = np.array([constant.K1, constant.K2, constant.P1, constant.P2, constant.K3])
        self.mtx = np.array([[constant.FX, 0, constant.CX], [0, constant.FY, constant.CY], [0,0,1]])
                             
    def calibrate_image(self, image):
        h, w = image.shape[:2]

        newimg, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w, h), 1, (w, h))
        calibrated = cv2.undistort(image, self.mtx, self.dist, None, newimg)

        x, y, w, h = roi
        calibrated = calibrated[y:y + h, x:x + w]
        return calibrated

    def save_calibrated_image(self, image):
        calibrated_image = self.calibrate_image(image)
        output_filename = datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + ".jpg"
        cv2.imwrite(output_filename, calibrated_image)
        cv2.destroyAllWindows()
        return output_filename

# Example usage
def example():
        calibrator = CameraCalibrator()
        #calibrator.calibrate_image(image)
        calibrated_image_path = calibrator.save_calibrated_image("./Frame.jpg") # 파일로 저장        # print("Calibrated image saved as:", calibrated_image_path)


class ColorDetector:
    def __init__(self, ip, username, password):
        self.ssh = paramiko.SSHClient()
        self.ssh.load_system_host_keys()
        self.ssh.connect(ip, 22, username, password)
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.scp = scp.SCPClient(self.ssh.get_transport())
        
        self.calibration = CameraCalibrator()

        self.lower_yellow = np.array([20, 100, 100])
        self.upper_yellow = np.array([40, 255, 255])
        self.lower_red1 = np.array([0, 50, 100])
        self.upper_red1 = np.array([20, 255, 255])
        self.lower_red2 = np.array([160, 50, 100])
        self.upper_red2 = np.array([179, 255, 255])
        self.lower_green = np.array([30, 50, 100])
        self.upper_green = np.array([90, 255, 255])
        self.lower_blue = np.array([100, 80, 100])
        self.upper_blue = np.array([130, 255, 255])

        self.kernel = np.ones((5, 5), np.uint8)
        
        self.cx_yellow = None
        self.cy_yellow = None
        self.cx_green = None
        self.cy_green = None
        self.cx_red = None
        self.cy_red = None
        self.yaw = None
        self.blue_centers = []
        
    def apply_CLAHE(self,bgr_img):
        lab_img = cv2.cvtColor(bgr_img,cv2.COLOR_BGR2Lab)
        L,a,b = cv2.split(lab_img)
        
        clahe = cv2.createCLAHE(clipLimit=3.0,tileGridSize=(8,8))
        L_histogram_equalized = clahe.apply(L)
        equalized_img = cv2.merge((L_histogram_equalized,a,b))
        equalized_img = cv2.cvtColor(equalized_img,cv2.COLOR_Lab2BGR)
        return equalized_img    

    def detect_colors(self):
        while True:
            self.scp.get('cam_image.jpg')
            # self.scp.get('./sangho/image.jpg')
            
            image = cv2.imread('cam_image.jpg')
            calibrated_image = self.calibration.calibrate_image(image=image)
            
            # equalized_image = self.apply_CLAHE(calibrated_image)
            
            
            hsv = cv2.cvtColor(calibrated_image, cv2.COLOR_BGR2HSV)

            mask_blue = cv2.inRange(hsv, self.lower_blue, self.upper_blue)
            mask_blue = cv2.erode(mask_blue, self.kernel, iterations=1)
            mask_blue = cv2.dilate(mask_blue, self.kernel, iterations=1)

            

            # 파란색 점들의 좌표 추출
            blue_points = np.argwhere(mask_blue == 255)
            
            if len(blue_points) > 0:
                # K-means 클러스터링을 사용하여 4개의 중심좌표 계산
                               
                kmeans = KMeans(n_clusters=4, random_state=0).fit(blue_points)
                self.blue_centers = np.array([[c[1], c[0]] for c in kmeans.cluster_centers_])

                # 파란색 중심점들 사이의 거리가 50 이하인 경우 빈 리스트로 할당
                if np.any(np.linalg.norm(np.diff(self.blue_centers, axis=0), axis=1) <= 50):
                    self.blue_centers = []
                    continue
                else:
                    # 파란색 중심점들을 왼쪽 위, 오른쪽 위, 오른쪽 아래, 왼쪽 아래 순서로 정렬
                    self.blue_centers = sorted(self.blue_centers, key=lambda x: (x[1], x[0]))
                    self.blue_centers[2:] = sorted(self.blue_centers[2:], key=lambda x: (-x[1], x[0]))

                    # 4개의 파란색 중심점 좌표 설정 (좌상, 우상, 우하, 좌하 순서)
                    pts = np.array(self.blue_centers, dtype=np.float32)
                    
            #     #보정
                pts[0,0] -= 2
                pts[0,1] -= 2
                pts[1,0] += 2
                pts[1,1] -= 2
                pts[2,0] += 2
                pts[2,1] += 2
                pts[3,0] -= 2
                pts[3,1] += 2

                # 결과 이미지의 꼭짓점 좌표 설정
                width, height = 320, 240
                dst_pts = np.array([(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)], dtype=np.float32)

                # 변환 행렬 계산
                M = cv2.getPerspectiveTransform(pts, dst_pts)
                
                # path.txt 파일 읽기
                with open('parkingtestmap.txt', 'r') as f:
                    path_data = f.read().strip()
                    path_points = []
                    # 문자열을 리스트로 변환
                    path_list = ast.literal_eval(path_data)
                    for point in path_list:
                        x, y = int(point[0]), int(point[1])
                        path_points.append((x, y))
                

                # 변환 행렬 적용하여 이미지 변환
                result = cv2.warpPerspective(calibrated_image, M, (width, height))

                # perspective transform 수행
                result = cv2.warpPerspective(calibrated_image, M, (width, height))

                # perspective transform 된 이미지에서 초록색과 빨간색 검출
                hsv_result = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)

                mask_yellow_result = cv2.inRange(hsv_result, self.lower_yellow, self.upper_yellow)
                mask_yellow_result = cv2.erode(mask_yellow_result, self.kernel, iterations=1)
                mask_yellow_result = cv2.dilate(mask_yellow_result, self.kernel, iterations=1)

                mask_red1_result = cv2.inRange(hsv_result, self.lower_red1, self.upper_red1)
                mask_red2_result = cv2.inRange(hsv_result, self.lower_red2, self.upper_red2)
                mask_red_result = cv2.bitwise_or(mask_red1_result, mask_red2_result)
                mask_red_result = cv2.erode(mask_red_result, self.kernel, iterations=1)
                mask_red_result = cv2.dilate(mask_red_result, self.kernel, iterations=1)

                mask_green_result = cv2.inRange(hsv_result, self.lower_green, self.upper_green)
                mask_green_result = cv2.erode(mask_green_result, self.kernel, iterations=1)
                mask_green_result = cv2.dilate(mask_green_result, self.kernel, iterations=1)

                moments_red_result = cv2.moments(mask_red_result)
                moments_green_result = cv2.moments(mask_green_result)

                if moments_red_result["m00"] != 0 and moments_green_result["m00"] != 0:
                    self.cx_red = int(moments_red_result["m10"] / moments_red_result["m00"])
                    self.cy_red = int(moments_red_result["m01"] / moments_red_result["m00"])
                    self.cx_green = int(moments_green_result["m10"] / moments_green_result["m00"])
                    self.cy_green = int(moments_green_result["m01"] / moments_green_result["m00"])

                    # result 이미지에 빨간색과 초록색 중심 좌표 표시
                    cv2.circle(result, (self.cx_red, self.cy_red), 2, (0, 0, 255), -1)
                    cv2.circle(result, (self.cx_green, self.cy_green), 2, (0, 255, 0), -1)

                    # result 이미지에서 빨간색과 초록색 중심을 잇는 파란색 선 그리기
                    cv2.line(result, (self.cx_red, self.cy_red), (self.cx_green, self.cy_green), (255, 0, 0), 2)

                    # result 이미지에서의 yaw 계산
                    yaw_result = math.atan2(self.cy_red - self.cy_green, self.cx_green - self.cx_red)
                    if yaw_result > math.pi:
                        yaw_result -= 2 * math.pi
                    print("Heading Angle in transformed image: {:.2f} radians".format(yaw_result))
                    print("Red center in transformed image - x: {}, y: {}".format(self.cx_red, self.cy_red))
                    print("Green center in transformed image - x: {}, y: {}".format(self.cx_green, self.cy_green))
                    print('')

                # ...

                # result 이미지에 path.txt의 점들을 빨간색으로 그리기
                for point in path_points:
                    cv2.circle(result, (round(point[0]),240 - round(point[1])), 2, (0, 0, 255), -1)

                cv2.imshow('Result', result)                
    
                # 중심좌표에 파란색 원 그리기
                for center in self.blue_centers:
                    cv2.circle(calibrated_image, (int(center[0]), int(center[1])), 2, (255, 0, 0), -1)
                    # print("Blue center x: {}, y: {}".format(int(center[0]), int(center[1])))    

            cv2.circle(calibrated_image, (0, 0), 2, (255, 255, 255), -1)

            if moments_red_result["m00"] != 0 and moments_green_result["m00"] != 0:
                cv2.line(calibrated_image, (self.cx_red, self.cy_red), (self.cx_green, self.cy_green), (255, 0, 0), 2)

                self.yaw = math.atan2(self.cy_red - self.cy_green, self.cx_green - self.cx_red)
                if self.yaw > math.pi:
                    self.yaw -= 2 * math.pi
                print("Heading Angle: {:.2f} radians".format(self.yaw))
                print('')

            cv2.imshow("Frame", calibrated_image)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

        cv2.destroyAllWindows()
        
    def get_yaw(self):
        return self.yaw
    
    def get_blue_centers(self):
        return self.blue_centers
    
    

if __name__ == '__main__':
    # 사용 예시
    detector = ColorDetector('192.168.0.5', 'pi', 'raspberry')
    detector.detect_colors()
    yaw = detector.get_yaw()
    blue_centers = detector.get_blue_centers()