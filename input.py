import cv2
import os
import imgaug.augmenters as iaa

def create_person_folder(name):
    base_path = 'data'
    person_path = os.path.join(base_path, name)
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    if not os.path.exists(person_path):
        os.makedirs(person_path)
    return person_path

def capture_images(name, num_images=5):
    person_path = create_person_folder(name)
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    print(f"Capturing {num_images} images for {name}. Press the spacebar to capture an image.")
    print("Press 'q' at any time to quit.")
    
    images_captured = 0
    
    # Define the augmentation sequence
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),  # horizontal flips
        iaa.Crop(percent=(0, 0.1)),  # random crops
        iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),  # blur images with a sigma of 0 to 0.5
        iaa.LinearContrast((0.75, 1.5)),  # improve or worsen the contrast
        iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)),  # add gaussian noise
        iaa.Multiply((0.8, 1.2)),  # make images darker or brighter
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},  # scale images to 80-120% of their size, individually per axis
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # translate by -20 to +20 percent (per axis)
            rotate=(-25, 25),  # rotate by -25 to +25 degrees
            shear=(-8, 8)  # shear by -8 to +8 degrees
        )
    ])
    
    while images_captured < num_images:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break
        
        cv2.imshow("Capture", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # Spacebar to capture image
            # Apply augmentations
            frame_aug = seq(image=frame)
            
            image_filename = os.path.join(person_path, f"image{images_captured+1}.jpg")
            cv2.imwrite(image_filename, frame_aug)
            print(f"Captured image {images_captured+1}/{num_images}")
            images_captured += 1
        elif key == ord('q'):  # 'q' to quit
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"Captured {images_captured} images for {name}.")

def save_images(image_list, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    for image in image_list:
        # Generate a unique filename
        existing_files = os.listdir(directory)
        new_filename = f"image_{len(existing_files) + 1}.jpg"
        
        # Save the image
        image_path = os.path.join(directory, new_filename)
        image.save(image_path)

def main():
    print('''
    This program captures images of your face for a face recognition system.
          
    - For accuracy, please ensure you are capturing multiple images from different angles, expressions, lighting conditions, and distances.
    - Best practice is to capture well-lit faces, but it is not required.
    - Its better to have your face in the center of the screen, but the model can detect faces off-center.
    - More images are better for accuracy, but 1-5 images should be enough.
          
    However, there is no need to follow this strictly. The model is pretty smart!  
    Note: press space to capture an image, and q to quit.
    ''')
    name = input("Please enter your name: ").strip()
    if not name:
        print("Error: Name cannot be empty.")
        return
    
    num_images = 5
    try:
        num_images = int(input("How many images would you like to capture? (default: 5): ") or 5)
    except ValueError:
        print("Invalid input. Using default value of 5 images.")
    
    capture_images(name, num_images)

if __name__ == "__main__":
    main()