#!/usr/bin/python
# -*- encoding: utf-8 -*-

import config
from face_parser.modelConfig import BiSeNet

import torch

import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
from skimage.color import rgb2lab
from skimage.segmentation import slic, mark_boundaries
from skimage.color import rgb2lab
import matplotlib.pyplot as plt



def check_sunglasses_opencv(image, glasses_mask):
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Create a mask for the glasses region
    glasses_region = cv2.bitwise_and(hsv_image, hsv_image, mask=glasses_mask)

    # Extract the V (brightness) and H (hue) channels
    v_channel = glasses_region[:, :, 2]

    # Get the brightness and hue values in the glasses region
    brightness_values = v_channel[glasses_mask == 1]

    # Calculate the average and standard deviation of brightness in the glasses region
    mean_brightness = np.mean(brightness_values)

    return mean_brightness < 120

class FaceParser:
    def __init__(self):
        """
            Initializes the face parser with the specified model path.
        """
        cp='79999_iter.pth'
        n_classes = 19
        self.net = BiSeNet(n_classes=n_classes)
        #net.cuda()
        current_directory = os.path.dirname(os.path.realpath(__file__))  # Get current directory
        save_pth = os.path.join(current_directory, 'res', 'cp', cp)
        self.net.load_state_dict(torch.load(save_pth, map_location=torch.device('cpu')))
        self.net.eval()
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    
    def vis_parsing_maps(self, im, parsing_anno, stride, save_im, save_path):
        '''
        function that is used to visualize the parsing
        '''
        # Colors for all 20 parts
        part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                    [255, 0, 85], [255, 0, 170],
                    [0, 255, 0], [85, 255, 0], [170, 255, 0],
                    [0, 255, 85], [0, 255, 170],
                    [0, 0, 255], [85, 0, 255], [170, 0, 255],
                    [0, 85, 255], [0, 170, 255],
                    [255, 255, 0], [255, 255, 85], [255, 255, 170],
                    [255, 0, 255], [255, 85, 255], [255, 170, 255],
                    [0, 255, 255], [85, 255, 255], [170, 255, 255]]

        im = np.array(im)
        vis_im = im.copy().astype(np.uint8)
        vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
        vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
        vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

        num_of_class = np.max(vis_parsing_anno)

        for pi in range(1, num_of_class + 1):
            index = np.where(vis_parsing_anno == pi)
            vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

        vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
        # print(vis_parsing_anno_color.shape, vis_im.shape)
        vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

        # Save result or not
        #print(save_im)
        if save_im:
            #cv2.imwrite(save_path[:-4] +'.png', vis_parsing_anno)
            cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        # return vis_im


    def parse_and_save_faces(self, image_path, output_path=None):
        with torch.no_grad():
            img = Image.open(image_path)
            image = img.resize((512, 512), Image.BILINEAR)
            img = self.to_tensor(image)
            img = torch.unsqueeze(img, 0)
            out = self.net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)

            filename, ext = os.path.splitext(os.path.basename(image_path))  # Extract filename and extension
            output_filename = filename + "_parsing" + ext

            if output_path:
                try:
                    # Create the output directory if it doesn't exist
                    os.makedirs(output_path, exist_ok=True)
                    # Save with rectangle in the provided output_path (directory)
                    faces_dir = output_path
                except OSError as e:
                    print(f"Error creating output directory: {e}")
                    # Handle the error gracefully, potentially use default saving behavior
            else:
                # Save with rectangle in "faces_detected" folder in script's directory
                script_dir = os.path.dirname(os.path.realpath(__file__))  # Get script's directory
                faces_dir = os.path.join(script_dir, "parsing_results")
                os.makedirs(faces_dir, exist_ok=True)  # Create "faces_detected" if it doesn't exist
            
            output_image_path = os.path.join(faces_dir, output_filename)
            self.vis_parsing_maps(image, parsing, stride=1, save_im=True, save_path=output_image_path)


    def has_glasses(self, image_path):
        with torch.no_grad():
            img = Image.open(image_path)
            image = img.resize((512, 512), Image.BILINEAR)
            img = self.to_tensor(image)
            img = torch.unsqueeze(img, 0)
            out = self.net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)

            has_glasses = np.any(parsing == 6)

            return has_glasses

    def has_sunglasses(self, image_path):
        with torch.no_grad():
            # Open and process the image
            img = Image.open(image_path)
            image = img.resize((512, 512), Image.BILINEAR)
            img_tensor = self.to_tensor(image)
            img_tensor = torch.unsqueeze(img_tensor, 0)

            # Get the parsing (segmentation map)
            out = self.net(img_tensor)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)

            # Check if sunglasses are detected (class 6 represents sunglasses)
            has_glasses = np.any(parsing == 6)

            if has_glasses:
                # Extract sunglasses region (class 6)
                sunglasses_mask = (parsing == 6).astype(np.uint8)

                # Convert original image to numpy array (for OpenCV processing)
                image_np = np.array(image)

                # Ensure the image has 3 channels (in case it's grayscale)
                if image_np.ndim == 2:
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)

                # Apply the mask to isolate sunglasses region
                sunglasses_pixels = cv2.bitwise_and(image_np, image_np, mask=sunglasses_mask)

                # Convert the region to HSV
                hsv_image = cv2.cvtColor(sunglasses_pixels, cv2.COLOR_BGR2HSV)

                # Extract the V (value) channel
                v_channel = hsv_image[:, :, 2]

                # Calculate the histogram of the V channel (Brightness)
                v_hist = cv2.calcHist([v_channel], [0], None, [256], [0, 256])
                v_hist = v_hist / v_hist.sum()

                less_dark = v_hist[20:50].sum()  # Slightly darker pixels
                #print(less_dark)

                if less_dark > config.MAX_LIGHT_DARK_SUN:
                    return True
                else:
                    return False
            else:
                return False

        
    def has_hat(self, image_path):
        with torch.no_grad():
            img = Image.open(image_path)
            image = img.resize((512, 512), Image.BILINEAR)
            img = self.to_tensor(image)
            img = torch.unsqueeze(img, 0)
            out = self.net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)

            has_hat = np.any(parsing == 18)

            return has_hat
            

    def detect_background(self, image_path):
        with torch.no_grad():
            img = Image.open(image_path)
            image = img.resize((512, 512), Image.BILINEAR)
            img = self.to_tensor(image)
            img = torch.unsqueeze(img, 0)
            
            # Forward pass through the network
            out = self.net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            
            return self.calculate_background(parsing, image)
        
    def detect_head_features(self, image_path):
        with torch.no_grad():
            img = Image.open(image_path)
            image = img.resize((512, 512), Image.BILINEAR)
            img = self.to_tensor(image)
            img = torch.unsqueeze(img, 0)
            #img = img.cuda()
            out = self.net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            # Convert the PIL image to a numpy array

            image = np.array(image)

            top_line = parsing[0]
            head_not_contained = np.any(top_line == 1)

            bottom_line = parsing[-1]
            chin_not_contained = np.any(bottom_line == 1)

            shoulder_check = self.calculation_shoulder_check(parsing)

            head_width, head_height = self.calculation_head_dimensions(parsing)

            return head_not_contained, chin_not_contained, head_width, head_height, shoulder_check
        

    def alternative_color_saturation(self, image_path):
        """
        Detects color saturation of a face in an image and saves the saturation histogram.

        Args:
            image_path (str): Path to the image file.

        Returns:
            str: "Over-saturated", "Under-saturated", or "Normal" saturation level.
        """
        with torch.no_grad():
            img = Image.open(image_path)
            image = img.resize((512, 512), Image.BILINEAR)
            img = self.to_tensor(image)
            img = torch.unsqueeze(img, 0)
            # img = img.cuda()  # Uncomment if using GPU

            image = np.array(image)


            out = self.net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)

            return self.calculate_saturation_personal(parsing, image)
        

        
    def detect_face_illumination(self, image_path):
        with torch.no_grad():
            img = Image.open(image_path)
            image = img.resize((512, 512), Image.BILINEAR)
            img = self.to_tensor(image)
            img = torch.unsqueeze(img, 0)
            # img = img.cuda()
            out = self.net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            
            return self.calculate_face_illumination(parsing, image)
        

    def face_contained(self, image_path):
        with torch.no_grad():
            img = Image.open(image_path)
            image = img.resize((512, 512), Image.BILINEAR)
            img = self.to_tensor(image)
            img = torch.unsqueeze(img, 0)
            #img = img.cuda()
            out = self.net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)

            top_line = parsing[0]
            head_not_contained = np.any(top_line == 1)

            bottom_line = parsing[-1]
            chin_not_contained = np.any(bottom_line == 1)

            return head_not_contained, chin_not_contained

    def head_dimension(self, image_path):
        with torch.no_grad():
            img = Image.open(image_path)
            image = img.resize((512, 512), Image.BILINEAR)
            img = self.to_tensor(image)
            img = torch.unsqueeze(img, 0)
            #img = img.cuda()
            out = self.net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)

            head_width, head_height = self.calculation_head_dimensions(parsing)

            return head_height, head_width
        
    def shoulder_check(self, image_path):
        with torch.no_grad():
            img = Image.open(image_path)
            image = img.resize((512, 512), Image.BILINEAR)
            img = self.to_tensor(image)
            img = torch.unsqueeze(img, 0)
            #img = img.cuda()
            out = self.net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)

            shoulder_check = self.calculation_shoulder_check(parsing)

            return shoulder_check 


    def detect_hat_glasses(self, image_path):
        with torch.no_grad():
            img = Image.open(image_path)
            image = img.resize((512, 512), Image.BILINEAR)
            img = self.to_tensor(image)
            img = torch.unsqueeze(img, 0)
            #img = img.cuda()
            out = self.net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            # Convert the PIL image to a numpy array

            has_glasses = np.any(parsing == 6)
            has_hat = np.any(parsing == 18)

            return has_glasses, has_hat


    def parser_analysis(self, image_path):
        with torch.no_grad():
            img = Image.open(image_path)
            image = img.resize((512, 512), Image.BILINEAR)
            img = self.to_tensor(image)
            img = torch.unsqueeze(img, 0)
            #img = img.cuda()
            out = self.net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            # Convert the PIL image to a numpy array

            image = np.array(image)

            top_line = parsing[0]
            head_not_contained = np.any(top_line == 1)

            bottom_line = parsing[-1]
            chin_not_contained = np.any(bottom_line == 1)

            has_glasses = np.any(parsing == 6)
            has_hat = np.any(parsing == 18)

            shoulder_check = self.calculation_shoulder_check(parsing)

            #momentarely not used
            #head_width, head_height = self.calculation_head_dimensions(parsing)

            color_saturation = self.calculate_saturation_personal(parsing, image)

            uniform_illumination = self.calculate_face_illumination(parsing, image)

            homogeneous_background = self.calculate_background(parsing, image)

            has_sunglasses = self.calculate_sunglasses(parsing, image)

            return has_hat, color_saturation, has_glasses, head_not_contained, chin_not_contained, shoulder_check, uniform_illumination, homogeneous_background, has_sunglasses
        

    def parser_list_analysis(self, image_path, checks=None):
        '''
        Executes specified controls and returns values for selected checks.
        If no checks are provided, all controls are executed.
        '''
        if checks is None:
            checks = ['has_hat', 'color_saturation', 'has_glasses', 
                    'head_not_contained', 'chin_not_contained', 
                    'shoulder_check', 'uniform_illumination', 'homogeneous_background', 'has_sunglasses']
        
        results = {}

        with torch.no_grad():
            img = Image.open(image_path)
            image = img.resize((512, 512), Image.BILINEAR)
            img = self.to_tensor(image)
            img = torch.unsqueeze(img, 0)
            # img = img.cuda()  # Uncomment if using GPU
            out = self.net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)

            # Convert the PIL image to a numpy array
            image = np.array(image)

            if 'head_not_contained' in checks:
                top_line = parsing[0]
                head_not_contained = np.any(top_line == 1)
                results['head_not_contained'] = head_not_contained

            if 'chin_not_contained' in checks:
                bottom_line = parsing[-1]
                chin_not_contained = np.any(bottom_line == 1)
                results['chin_not_contained'] = chin_not_contained

            if 'has_sunglasses' in checks:
                has_glasses = np.any(parsing == 6)
                if has_glasses:
                    has_sunglasses = self.calculate_sunglasses(parsing, image)
                    results['has_sunglasses'] = has_sunglasses
                else:
                    results['has_sunglasses'] = False

            if 'has_hat' in checks:
                has_hat = np.any(parsing == 18)
                results['has_hat'] = has_hat

            if 'shoulder_check' in checks:
                shoulder_check = self.calculation_shoulder_check(parsing)
                results['shoulder_check'] = shoulder_check

            if 'color_saturation' in checks:
                color_saturation = self.calculate_saturation_personal(parsing, image)
                results['color_saturation'] = color_saturation

            if 'uniform_illumination' in checks:
                uniform_illumination = self.calculate_face_illumination(parsing, image)
                results['uniform_illumination'] = uniform_illumination

            if 'homogeneous_background' in checks:
                homogeneous_background = self.calculate_background(parsing, image)
                results['homogeneous_background'] = homogeneous_background

        return results

    '''
    CALCULATION HELPER FUNCTIONS
    '''

    def calculate_background(self, parsing, image):
        # Convert the PIL image to a numpy array
        image_np = np.array(image)
        
        # Extract the background pixels (where parsing == 0)
        background_mask = (parsing == 0).astype(np.uint8)
        background_pixels = image_np[background_mask == 1]

        # Compute color variance
        variance = np.var(background_pixels, axis=0)

        average_variance = np.mean(variance)

        image_lab = rgb2lab(image_np)

        # Perform SLIC superpixel segmentation
        segments = slic(image_lab, n_segments=200, compactness=10, sigma=1, start_label=1)

        # Find the unique segments
        all_segments = np.unique(segments)

        # Only keep segments where ALL pixels belong to the background
        background_segments = []
        for segment in all_segments:
            segment_mask = (segments == segment)
            # Check if all pixels of the segment are background pixels
            if np.all(background_mask[segment_mask] == 1):
                background_segments.append(segment)

        # Compute the proportion of superpixels that are homogeneous
        homogeneous_count = 0
        for segment in background_segments:
            segment_mask = (segments == segment)
            segment_pixels = image_np[segment_mask]
            # Compute color variance within this segment
            segment_variance = np.var(segment_pixels, axis=0)
            if np.max(segment_variance) < config.SUPERPIXEL_VARIANCE_THRESHOLD:
                homogeneous_count += 1

        # Calculate the proportion of homogeneous background superpixels
        if len(background_segments) > 0:
            proportion_homogeneous = homogeneous_count / len(background_segments)
        else:
            proportion_homogeneous = 0  # Handle case where no valid background segments are found
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray_image, 30, 100)

        # Count number of edges in the background
        num_background_edges = np.sum(edges[background_mask])
        
        # Return the number of edges in the background
        return num_background_edges < config.MAX_EDGES_THRESHOLD and average_variance < config.AVG_VARIANCE_THRESHOLD and proportion_homogeneous > config.HOMOGENEOUS_PROPORTION_THRESHOLD
        


    def calculation_head_dimensions(self, parsing):
        mask = (parsing == 1).astype(np.uint8)  # Assuming 1 indicates a skin pixel
        left_ear_mask = (parsing == 7).astype(np.uint8)
        right_ear_mask = (parsing == 8).astype(np.uint8)

        left_ear_coords = np.column_stack(np.where(left_ear_mask == 1))
        right_ear_coords = np.column_stack(np.where(right_ear_mask == 1))

        # Find the middle x point of each ear
        if len(left_ear_coords) > 0:
            left_ear_mid_x = np.mean(left_ear_coords[:, 1])
        else:
            left_ear_mid_x = None

        if len(right_ear_coords) > 0:
            right_ear_mid_x = np.mean(right_ear_coords[:, 1])
        else:
            right_ear_mid_x = None

        face_coords = np.column_stack(np.where(mask == 1))
        # Compute the distance between the two points if both ears are present
        if left_ear_mid_x is not None and right_ear_mid_x is not None:
            head_width = abs(right_ear_mid_x - left_ear_mid_x)
        else:
            # If one or both ears are missing, compute the maximum width of the face pixels
            if len(face_coords) > 0:
                face_width = np.max(face_coords[:, 1]) - np.min(face_coords[:, 1])
            else:
                face_width = 0  # In case no face pixels are detected
            head_width = face_width

        # Compute the distance between the lowest point in the face and the highest
        if len(face_coords) > 0:
            head_height = np.max(face_coords[:, 0]) - np.min(face_coords[:, 0])
        else:
            head_height = 0  # In case no face pixels are detected

        return head_width, head_height
    

    def calculation_shoulder_check(self, parsing):
        #shoulder_mask
        shoulder_mask = (parsing == 16).astype(np.uint8)

        image_width = parsing.shape[1]
        image_center_x = image_width // 2

        # Get the coordinates of all shoulder pixels
        shoulder_pixels = np.argwhere(shoulder_mask == 1)

        # Split the shoulder pixels into left and right based on face_center_x
        left_shoulder_pixels = shoulder_pixels[shoulder_pixels[:, 1] < image_center_x]
        right_shoulder_pixels = shoulder_pixels[shoulder_pixels[:, 1] >= image_center_x]

        # Count the shoulder pixels in each half
        left_count = len(left_shoulder_pixels)
        right_count = len(right_shoulder_pixels)

        # Output the counts for debugging
        def calculate_centroid(pixels):
            if len(pixels) == 0:
                return (0, 0)  # Avoid division by zero
            return (np.mean(pixels[:, 0]), np.mean(pixels[:, 1]))

        left_centroid = calculate_centroid(left_shoulder_pixels)
        right_centroid = calculate_centroid(right_shoulder_pixels)

        shoulder_check = True

        pixel_ratio = min(left_count, right_count) / max(left_count, right_count) if max(left_count, right_count) > 0 else 0
        if pixel_ratio < config.MAX_SHOULDER_PIXEL_RATIO:
            shoulder_check = False

        if abs(left_centroid[0] - right_centroid[0]) > config.MAX_SHOULDER_Y_DISTANCE:
            shoulder_check = False

        return shoulder_check
    
    
    def calculate_saturation_personal(self, parsing, image):
        #face mask
        mask = (parsing == 1).astype(np.uint8)

        # Convert to OpenCV format (BGR)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Apply face mask
        masked_image = cv2.bitwise_and(image, image, mask=mask)

        # Convert masked image to HSV color space
        hsv_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2HSV)

        oversaturated_pixels = np.sum(hsv_image[:, :, 1] > 200)
        undersaturated_pixels = np.sum(hsv_image[:, :, 1] < 50)
        peak_pixels = np.sum(hsv_image[:, :, 1] < 5)
        total_pixels = hsv_image.shape[0] * hsv_image.shape[1]

        oversaturation_percentage = oversaturated_pixels / total_pixels * 100
        undersaturation_percentage = (undersaturated_pixels - peak_pixels) / total_pixels * 100

        #print(undersaturation_percentage)

        #return oversaturation_percentage, undersaturation_percentage

        bad_saturation = oversaturation_percentage > config.OVERSATURATION_THRESHOLD or undersaturation_percentage > config.UNDERSATURATION_THRESHOLD

        return bad_saturation
    
    def calculate_face_illumination(self, parsing, image):
        # Convert the PIL image to a numpy array
        image = np.array(image)
        
        # Create mask for face pixels
        mask = (parsing == 1).astype(np.uint8)
        
        # Apply the mask to the image to get only face pixels
        face_pixels = cv2.bitwise_and(image, image, mask=mask)
        
        face_pixels_gray = cv2.cvtColor(face_pixels, cv2.COLOR_BGR2GRAY)
        
        # Mask out the non-face areas in the grayscale image
        face_pixels_gray_masked = face_pixels_gray[mask == 1]

        bright_pixels = np.count_nonzero(face_pixels_gray_masked > 220)
        total_pixels = face_pixels_gray_masked.size
        bright_pixels_percentage = (bright_pixels / total_pixels) * 100

        dark_pixels = np.count_nonzero(face_pixels_gray_masked < 100)
        dark_pixels_percentage = (dark_pixels / total_pixels) * 100

        if bright_pixels_percentage > config.MAX_BRIGHT_LIGHT or dark_pixels_percentage > config.MAX_DARK_LIGHT:
            return False
        else:
            return True
        

    def calculate_sunglasses(self, parsing, image):
        # Check if sunglasses are detected (class 6 represents sunglasses)
        has_glasses = np.any(parsing == 6)
        if has_glasses:
            # Extract sunglasses region (class 6)
            sunglasses_mask = (parsing == 6).astype(np.uint8)

            # Convert original image to numpy array (for OpenCV processing)
            image_np = np.array(image)

            # Ensure the image has 3 channels (in case it's grayscale)
            if image_np.ndim == 2:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)

            # Apply the mask to isolate sunglasses region
            sunglasses_pixels = cv2.bitwise_and(image_np, image_np, mask=sunglasses_mask)

            # Convert the region to HSV
            hsv_image = cv2.cvtColor(sunglasses_pixels, cv2.COLOR_BGR2HSV)

            # Extract the V (value) channel
            v_channel = hsv_image[:, :, 2]

            # Calculate the histogram of the V channel (Brightness)
            v_hist = cv2.calcHist([v_channel], [0], None, [256], [0, 256])
            v_hist = v_hist / v_hist.sum()

            less_dark = v_hist[20:50].sum()  # Slightly darker pixels
            #print(less_dark)

            if less_dark > config.MAX_LIGHT_DARK_SUN:
                return True
            else:
                return False
        else:
            return False



