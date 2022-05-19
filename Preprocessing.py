import numpy as np
import cv2 as cv
import mat73
import scipy.ndimage
dim = (256,256)
NOISE_FLOOR = 20
data_path = 'C:\\Users\osman\Desktop\YSA_Proje\data\\'
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
   
def reduce_noise(noisy_img, noise_floor=NOISE_FLOOR):
        clean_img = np.array(noisy_img)
        clean_img[noisy_img < NOISE_FLOOR] = 0
        return clean_img
        


def siyah_kenar(img):
    x,y = img.shape[0],img.shape[1]
    minpix = 20 # minimum piksel değeri
    x_r = []
    x_l = []
    y_up = []
    y_down = []
    for j in range(int(x/2)):
            for i in range(y):
                if img[i][j] <0:
                    img[i][j] =0
                if img[i][j] > minpix :
                    x_l.append(j)
                    break 

    for j in range(int((x/2)-1),x):
        for i in range(y):
            if img[i][j] <0:
                img[i][j] =0
            if img[i][j] > minpix :
                x_r.append(j)
                break
               

    for i in range(int((y/2)-1)):
            for j in range(x):
                if img[i][j] <0:
                    img[i][j] =0
                if img[j][i] > minpix :
                    y_up.append(j)
                    break 

    for i in range(int(y/2),y):
        for j in range(x):
            if img[i][j] <0:
                img[i][j] =0
            if img[j][i] > minpix :
                y_down.append(j)
                break
                
    xmin = np.amin(x_l)
    xmax = np.amax(x_r)  
    
    ymin = np.amin(y_up)
    ymax = img.shape[0]-np.amin(y_down)

    # print(xmin, xmax, ymin,ymax )
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
    
def scale_img(img):
    mx = np.amax(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):        
            img[i][j] *= 255.0 / mx
   
    return img

def resize(img):
    img = cv.resize(img, dim, interpolation = cv.INTER_AREA)
    return img
                
def preprocesss(input_img):
    output_img = reduce_noise(input_img)
    output_img = scale_img(input_img)
    output_img = siyah_kenar(output_img)
    # output_img = cv.morphologyEx(output_img, cv.MORPH_GRADIENT, kernel)
    output_img = resize(output_img)
    #     # mask = cv.resize(mask, dim, interpolation=cv.INTER_AREA)
    
    return output_img 

# img = mat73.loadmat(data_path+ '999.mat')['cjdata']['image'] 

# img = preprocesss(img)

# plt.imshow(img, cmap = 'gray')    
            
            

  
    
  
        