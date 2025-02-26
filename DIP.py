from PIL import Image
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np

def load_image(file_path):
    try:
        img = Image.open(file_path)  
        #plt.imshow(img)  
        #plt.axis('off')
        #plt.show()

        
        grayscale_img = img.convert('L') 
        
        #plt.imshow(grayscale_img, cmap='gray')  
        #plt.axis('off')
        #plt.show()
        grayscale_img.save(file_path)
        return grayscale_img
    except Exception as e:
        print(f"Error: {e}")
        return None


def dir_crawler(dataset_path):
    try:
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                _, file_ext = os.path.splitext(file)
                if file_ext.lower() == ".jpg":  
                    file_path = os.path.join(root, file)
                    print(f"Processing: {file_path}")
                    load_image(file_path) 
    except Exception as e:
        print(f"Error during directory crawling: {e}")


dataset_path = r"C:\Users\Desert Fox\Documents\dataset\dataset\A"
background1 = r"C:\Users\Desert Fox\Documents\dataset\dataset\A\nothing1.jpg"
background2 = r"C:\Users\Desert Fox\Documents\dataset\dataset\A\nothing2.jpg"
background3 = r"C:\Users\Desert Fox\Documents\dataset\dataset\A\nothing3.jpg"
ex_img = r"C:\Users\Desert Fox\Documents\dataset\dataset\A\A11.jpg"
#dir_crawler(dataset_path)
#load_image(dataset_path)


backgrounds = [
    np.array(Image.open(background1)), 
    np.array(Image.open(background2)),
    np.array(Image.open(background3))
]

   
def compare_images(img, backgrounds):
    min_diff = float('inf')
    best_match = None

    for bg in backgrounds:
        diff = np.sum(np.abs(img.astype(np.int16) - bg.astype(np.int16)))  # Compute sum of absolute differences
        if diff < min_diff:
            min_diff = diff
            best_match = bg

    return best_match


def process_image(image_path, backgrounds):
    
    img = np.array(load_image(image_path))  
    
    matching_bg = compare_images(img, backgrounds)
    
    if matching_bg is not None:
        #result = np.clip(img.astype(np.int16) - matching_bg.astype(np.int16), 0, 255).astype(np.uint8)
        #result_image = Image.fromarray(result)
        
        
        #result_image.save(image_path)

        diff = np.abs(img.astype(np.float32) - matching_bg.astype(np.float32))

        
        diff = (diff - diff.min()) / (diff.max() - diff.min()) * 255  

        result = diff.astype(np.uint8)
        result_image = Image.fromarray(result)
        result_image.save(image_path)
        
    else:
        print(f"No matching background found for {image_path}")



process_image(ex_img, backgrounds)



def classifier_cnn(img_rows=200, img_cols=200, num_classes=28):
    model = Sequential()
    model.add(Conv2D(32, (3,3), activation='relu', input_shape=(img_rows, img_cols, 1), padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.5))  
    model.add(Dense(num_classes, activation='softmax'))

    return model


num_classes = 28  
model = classifier_cnn(img_rows=200, img_cols=200, num_classes=num_classes)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
