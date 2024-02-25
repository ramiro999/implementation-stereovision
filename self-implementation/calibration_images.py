import cv2
import matplotlib.pyplot as plt

# Initialize video capture for both cameras
cap = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)

num = 0

# Ensure both cameras are opened
while cap.isOpened() and cap2.isOpened():
    success1, img = cap.read()
    success2, img2 = cap2.read()

    # Check if images were successfully read
    if success1 and success2:
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        # Display images using Matplotlib
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(img_rgb)
        plt.title('Img 1')
        plt.axis('off')  # Hide axis
        
        plt.subplot(1, 2, 2)
        plt.imshow(img2_rgb)
        plt.title('Img 2')
        plt.axis('off')  # Hide axis
        
        plt.show(block=False)
        plt.pause(0.001)  # Pause to ensure the figure is displayed correctly
        
        k = cv2.waitKey(5)
    # if k == 27:  # ESC key to exitopen
        # break
    # elif k == ord('s'):  # 's' key to save and exit
        cv2.imwrite('images/stereoLeft/imageL' + str(num) + '.png', img)
        cv2.imwrite('images/stereoRight/imageR' + str(num) + '.png', img2)
        print("Images saved!")
        num += 1

# Release and destroy all windows before termination
cap.release()
cap2.release()
cv2.destroyAllWindows()
plt.close('all')  # Close all Matplotlib windows
