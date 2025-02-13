from PIL import Image
import matplotlib.pyplot as plt
import os


def load_image(file_path):
    try:
        img = Image.open(file_path)  
        plt.imshow(img)  
        plt.axis('off')
        plt.show()

        
        grayscale_img = img.convert('L') 
        plt.imshow(grayscale_img, cmap='gray')  
        plt.axis('off')
        plt.show()
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
                if file_ext.lower() == ".png":  
                    file_path = os.path.join(root, file)
                    print(f"Processing: {file_path}")
                    load_image(file_path) 
    except Exception as e:
        print(f"Error during directory crawling: {e}")


dataset_path = r"C:\Users\Desert Fox\Documents\dataset\dataset\A"


#load_image(dataset_path)

