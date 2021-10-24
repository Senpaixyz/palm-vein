from flask import Flask,render_template, Response, request
from flask_ngrok import run_with_ngrok
import os
from PIL import Image
import base64
import io
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import cv2
from sewar import full_ref
from skimage import measure, metrics
from numpy.fft import fft2, ifft2
from scipy.signal import convolve2d
from skimage.metrics import structural_similarity as ssim
from skimage import color, data, restoration
import base64
import io
matplotlib.use('Agg')
app = Flask(__name__)
#run_with_ngrok(app)
UPLOAD_FOLDER = 'static/upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 3024 * 3024
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png']


class Filters(object):

    def __init__(self, image):
        self.roi = image
        self.gaussian_kernel = np.ones((5,5),np.float32)/25
        self.mean_kernel = np.ones((3,3),np.float32)/25
        self.gabor_kernel = self.get_gabor_kernel()
        self.homomorphic_kernel = self.get_homomorphic_filter()


    def gaussian_filter(self):
        conv_gaussian = cv2.filter2D(self.roi, -1, self.gaussian_kernel)
        return conv_gaussian

    def wiener_filter(self):
        # Apply Wiener Filter
        img = np.copy(self.roi)
        gray_img = color.rgb2gray(img)
        psf = np.ones((5, 5)) / 25
        img = convolve2d(gray_img, psf, 'same')
        img += 0.1 * img.std() * np.random.standard_normal(img.shape)
        deconvolved_img = restoration.wiener(img, psf, 3)

        return deconvolved_img

    def mean_filter(self):
        conv_mean = cv2.filter2D( self.roi, -1, self.mean_kernel)
        return conv_mean

    def get_gabor_kernel(self):
        ksize = 3  # Use size that makes sense to the image and fetaure size. Large may not be good.
        # On the synthetic image it is clear how ksize affects imgae (try 5 and 50)
        sigma = 3  # Large sigma on small features will fully miss the features.
        theta = 1 * np.pi / 2  # /4 shows horizontal 3/4 shows other horizontal. Try other contributions
        lamda = 1 * np.pi / 4  # 1/4 works best for angled.
        gamma = 0.9  # Value of 1 defines spherical. Calue close to 0 has high aspect ratio
        # Value of 1, spherical may not be ideal as it picks up features from other regions.
        phi = 0  # Phase offset. I leave it to 0. (For hidden pic use 0.8)

        gabor_kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, phi, ktype=cv2.CV_32F)
        return gabor_kernel

    def get_homomorphic_filter(self):
        # take ln of image
        img_log = np.log(np.float64(self.roi), dtype=np.float64)

        # do dft saving as complex output
        dft = np.fft.fft2(img_log, axes=(0, 1))

        # apply shift of origin to center of image
        dft_shift = np.fft.fftshift(dft)

        # create black circle on white background for high pass filter
        # radius = 3
        radius = 9
        mask = np.zeros_like(self.roi, dtype=np.float64)
        cy = mask.shape[0] // 2
        cx = mask.shape[1] // 2
        cv2.circle(mask, (cx, cy), radius, 1, -1)
        mask = 1 - mask

        # antialias mask via blurring
        # mask = cv2.GaussianBlur(mask, (9,9), 0)
        mask = cv2.GaussianBlur(mask, (47, 47), 0)

        # apply mask to dft_shift
        dft_shift_filtered = np.multiply(dft_shift, mask)

        # shift origin from center to upper left corner
        back_ishift = np.fft.ifftshift(dft_shift_filtered)

        # do idft saving as complex
        img_back = np.fft.ifft2(back_ishift, axes=(0, 1))

        # combine complex real and imaginary components to form (the magnitude for) the original image again
        img_back = np.abs(img_back)

        # apply exp to reverse the earlier log
        img_homomorphic = np.exp(img_back, dtype=np.float64)

        # scale result
        img_homomorphic = cv2.normalize(img_homomorphic, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                        dtype=cv2.CV_8U)

        return img_homomorphic


    def gabor_filter(self):
        conv_gabor = cv2.filter2D(self.roi, -1, self.gabor_kernel)
        return conv_gabor

    def homomorphic_filter(self):
        return self.homomorphic_kernel



@app.route('/')
def Home():
    return render_template('/index.html',
                           is_error=False,
                           error='',
                           is_uploaded=False,
                           img_path='',
                           )
@app.route('/display' ,methods=['GET','POST'])
def display():
    is_uploaded = False
    if request.method == 'POST':
        files = request.files.getlist("images[]")
        paths = []
        max_upload = 3
        i = 0
        size = 304, 352
        for file in files:
            img = file.stream
            if i < max_upload:
                img_size = len(img.read())
                img.seek(0)
                image_string_encoded = base64.b64encode(img.read())
                image_string = image_string_encoded.decode('utf-8')
                img.seek(0)

                imgIO = io.BytesIO(img.read())
                imgIO.seek(0)
                image_IO = Image.open(imgIO)
                img_width = image_IO.size[0]
                img_height = image_IO.size[1]
                buffered = io.BytesIO()
                buffered.seek(0)
                image_IO = image_IO.resize((289, 289))
                image_IO.save(buffered, format="PNG")
                image_string_encoded = base64.b64encode(buffered.getvalue())
                image_string = image_string_encoded.decode('utf-8')


                imgIO.seek(0)
                # set img display screen to resize the display
                imgIO_display = image_IO
                buffered2 = io.BytesIO()
                buffered2.seek(0)
                imgIO_display = imgIO_display.resize((400, 400))
                imgIO_display.save(buffered2, format="PNG")
                img_display_string_encoded = base64.b64encode(buffered2.getvalue())
                img_display_string = img_display_string_encoded.decode('utf-8')
                imgIO_display.seek(0)
                new_filename = 'Palm-Image[' + str(i+1) + '].png'
                img_info = {
                    'path': img_display_string,
                    'filename': new_filename,
                    'img_string': image_string,
                    'width': img_width,
                    'height': img_height,
                    'size': img_size
                }
                paths.append(img_info)
            i += 1
        print(paths)
        is_uploaded = True
        print(paths)
        return render_template('/display.html',
                               is_uploaded= is_uploaded,
                               img_path=paths,
                               )

def decode_image_string(image_array):
    plt.imshow(image_array, cmap='gray')
    plt.axis('off')
    buffered = io.BytesIO()
    plt.savefig(buffered, bbox_inches='tight', pad_inches=0, format='PNG')
    buffered.seek(0)
    image_string_encoded = base64.b64encode(buffered.getvalue())
    image_string = image_string_encoded.decode('utf-8')
    return image_string

@app.route('/predict' ,methods=['GET','POST'])
def predict():
    is_predicted = False
    if request.method == 'POST':
        try:
            base64_str = request.form['img_path']
            img_o = plt.imread(io.BytesIO(base64.decodebytes(bytes(base64_str, "utf-8"))), 1)
            base64_str_img = cv2.resize(img_o, (289, 289))

            img = base64_str_img
            filters = Filters(img)
            # all image string...
            gaussian = decode_image_string(filters.gaussian_filter())
            gabor = decode_image_string(filters.gabor_filter())
            mean = decode_image_string(filters.mean_filter())
            homomorphic = decode_image_string(filters.homomorphic_filter())
            wiener = decode_image_string(filters.wiener_filter())

            # decode image string for comparison to original image
            gaussian_img = plt.imread(io.BytesIO(base64.decodebytes(bytes(gaussian, "utf-8"))), 1)
            gaussian_resize = cv2.resize(gaussian_img, (289, 289))

            gabor_img = plt.imread(io.BytesIO(base64.decodebytes(bytes(gabor, "utf-8"))), 1)
            gabor_resize = cv2.resize(gabor_img, (289, 289))

            mean_img = plt.imread(io.BytesIO(base64.decodebytes(bytes(mean, "utf-8"))), 1)
            mean_resize = cv2.resize(mean_img, (289, 289))

            homomorphic_img = plt.imread(io.BytesIO(base64.decodebytes(bytes(homomorphic, "utf-8"))), 1)
            homomorphic_resize = cv2.resize(homomorphic_img, (289, 289))

            wiener_img = plt.imread(io.BytesIO(base64.decodebytes(bytes(wiener, "utf-8"))), 1)
            wiener_resize = cv2.resize(wiener_img, (289, 289))
            # compute for RMSE
            print("IMG: ", img.shape, " GAUSSIAN: ", gaussian_resize.shape)
            gaussian_rmse_skimg = metrics.normalized_root_mse(img, gaussian_resize)
            gabor_rmse_skimg = metrics.normalized_root_mse(img, gabor_resize)
            mean_rmse_skimg = metrics.normalized_root_mse(img, mean_resize)
            homomorphic_rmse_skimg = metrics.normalized_root_mse(img, homomorphic_resize)
            wiener_rmse_skimg = metrics.normalized_root_mse(img, wiener_resize)
            # compute for  SSIM
            gaussian_ssim_skimg = ssim(img, gaussian_resize, data_range=img.max() - img.min(), multichannel=True)
            gabor_ssim_skimg = ssim(img, gabor_resize, data_range=img.max() - img.min(), multichannel=True)
            mean_ssim_skimg = ssim(img, mean_resize, data_range=img.max() - img.min(), multichannel=True)
            homomorphic_ssim_skimg = ssim(img, homomorphic_resize, data_range=img.max() - img.min(), multichannel=True)
            wiener_ssim_skimg = ssim(img, wiener_resize, data_range=img.max() - img.min(), multichannel=True)

            # decode again
            normal_image = decode_image_string(base64_str_img)
            gaussian_img = decode_image_string(gaussian_resize)
            gabor_img = decode_image_string(gabor_resize)
            mean_img = decode_image_string(mean_resize)
            homomorphic_img = decode_image_string(homomorphic_resize)
            wiener_img = decode_image_string(wiener_resize)


            image_default = {
                    'name': 'Normal Palm Vein',
                    'string': normal_image
                }
            filtered_images = [
                    {
                        'name': 'Gaussian Filter',
                        'string': gaussian_img,
                        'rmse': gaussian_rmse_skimg,
                        'ssim': gaussian_ssim_skimg,
                        'description': 'Add description in this section'
                    },
                    {
                        'name': 'Gabor Filter',
                        'string': gabor_img,
                        'rmse': gabor_rmse_skimg,
                        'ssim': gabor_ssim_skimg,
                        'description': 'Add description in this section'
                    },
                    {
                        'name': 'Mean Filter',
                        'string': mean_img,
                        'rmse': mean_rmse_skimg,
                        'ssim': mean_ssim_skimg,
                        'description': 'Add description in this section'
                    },
                    {
                        'name': 'Homomorphic Filter',
                        'string': homomorphic_img,
                        'rmse': homomorphic_rmse_skimg,
                        'ssim': homomorphic_ssim_skimg,
                        'description': 'Add description in this section'
                    },
                    {
                        'name': 'Wiener Filter',
                        'string': wiener_img,
                        'rmse': wiener_rmse_skimg,
                        'ssim': wiener_ssim_skimg,
                        'description': 'Add description in this section'
                    },
                ]
            is_predicted = True
            return render_template('predict.html',
                                       is_predicted=is_predicted,
                                       filtered_images=filtered_images,
                                       normal_image=image_default
                                      )
        except ValueError as e:
            return render_template('index.html',
                                   is_error=True,
                                   error=e,
                                   is_uploaded=False,
                                   img_path='',
                                   )
@app.errorhandler(413)
def request_entity_too_large(error):
    msg = "File Too Large - " + str(error)
    return render_template('index.html',
                           error=error,
                           is_uploaded=False,
                           img_path='',
                           )

@app.errorhandler(404)
def request_entity_too_large(error):
    msg = str(error)
    return render_template('index.html',
                           error=error,
                           is_uploaded=False,
                           img_path='',
                           )
if __name__ == '__main__':
    app.run(debug=True, port=3500)
    #app.run()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
