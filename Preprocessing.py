import numpy as np
import cv2 as cv
import mat73
import scipy.ndimage
NOISE_FLOOR = 20
class Preprocess():
    import cv2 as cv
    global dimension 
    global NOISE_FLOOR
    global kernel
    global data_path
    dimension = (256,256)
    NOISE_FLOOR = 20
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
       
    def reduce_noise(noisy_img, noise_floor=NOISE_FLOOR):
            clean_img = np.array(noisy_img)
            clean_img[noisy_img < NOISE_FLOOR] = 0
            return clean_img
            
    def remove_black_padding(img, minpix_val = 20):
        x,y = img.shape[0],img.shape[1]
        x_axis_right_points = []
        x_axis_left_points = []
        y_axis_up_points = []
        y_axis_down_points = []
        for j in range(int(x/2)):
                for i in range(y):
                    if img[i][j] <0:
                        img[i][j] =0
                    if img[i][j] > minpix_val :
                        x_axis_left_points.append(j)
                        break 
    
        for j in range(int((x/2)-1),x):
            for i in range(y):
                if img[i][j] <0:
                    img[i][j] =0
                if img[i][j] > minpix_val :
                    x_axis_right_points.append(j)
                    break
                   
    
        for i in range(int((y/2)-1)):
                for j in range(x):
                    if img[i][j] <0:
                        img[i][j] =0
                    if img[j][i] > minpix_val :
                        y_axis_up_points.append(j)
                        break 
    
        for i in range(int(y/2),y):
            for j in range(x):
                if img[i][j] <0:
                    img[i][j] =0
                if img[j][i] > minpix_val :
                    y_axis_down_points.append(j)
                    break
                    
        xmin = np.amin(x_axis_left_points)
        xmax = np.amax(x_axis_right_points)  
        ymin = np.amin(y_axis_up_points)
        ymax = img.shape[0]-np.amin(y_axis_down_points)
          
        return img[ymin:ymax,xmin:xmax]
        
    def square_image(rect_img):
            X, Y = rect_img.shape
            if X != Y:
                if X > Y:
                    out_img = np.zeros((X, X))
                    offset = int(np.ceil((X-Y) / 2))
                    out_img[:, offset : (offset + Y)] = rect_img
                else:
                    out_img = np.zeros((Y, Y))
                    offset = int(np.ceil((Y-X) / 2))
                    out_img[offset : (offset + X), :] = rect_img
                return out_img
            else:
                return rect_img
        
    def scale_image(img):
        max_pix_value = np.amax(img)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):        
                img[i][j] *= 255.0 / max_pix_value      
        return img
    
    def resize_image(img):
        img = cv.resize(img, dimension, interpolation = cv.INTER_AREA)
        return img
                    
    def preprocessing(input_img, minpix_val = 20):
        output_img = Preprocess.reduce_noise(input_img)
        output_img = Preprocess.scale_image(input_img)
        output_img = Preprocess.remove_black_padding(output_img, minpix_val)
        output_img = Preprocess.cv.morphologyEx(output_img, cv.MORPH_GRADIENT, kernel)
        output_img = Preprocess.resize_image(output_img)
        return output_img       
