from ultralytics import YOLO
import cv2  as cv
import time
model = YOLO("best.pt")
# yolov8n.pt

cap = cv.VideoCapture(0)

while True:

    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame,conf=0.4,verbose=False)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()


        if len(results[0]) > 0:
         for result in results:
            if result.boxes:
                box = result.boxes[0]
                class_id = int(box.cls)
                object_name = model.names[class_id]
                print(object_name)
                
        # Display the annotated frame
        cv.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break
cap.release()
cv.destroyAllWindows()



# # from ultralytics import YOLO
# # import cv2 as cv

# # model = YOLO("best.pt")  # Replace with the path to your YOLO model file

# # cap = cv.VideoCapture(0)

# # while True:
# #     success, frame = cap.read()

# #     if success:
# #         # Run YOLOv8 inference on the frame
# #         results = model(frame, conf=0.01)

# #         # Extract class names from the results
# #         class_names = results[0][:,:].tolist() if results[0] is not None else []

# #         # Print detected class names to the terminal
# #         if class_names:
# #             print("Detected Classes:", class_names)

# #         # Break the loop if 'q' is pressed
# #         if cv.waitKey(1) & 0xFF == ord("q"):
# #             break
# #     else:
# #         # Break the loop if the end of the video is reached
# #         break

# # cap.release()
# # cv.destroyAllWindows()



# from ultralytics import YOLO
# import cv2 as cv

# # Load your YOLOv8 model (replace "best.pt" with your model path)
# model = YOLO("best.pt")

# # Load the image you want to detect from
# image_path = "test-2.png"  # Replace with your image path
# image = cv.imread(image_path)

# # Run YOLOv8 inference on the image
# results = model(image, conf=0.4)  # Adjust confidence threshold as needed

# # Annotate the image with bounding boxes and labels
# annotated_image = results[0].plot()  # Use results[0] to get the first result

# # Print detected object names
# if len(results[0]) > 0:
#     for result in results[0]:
#         if result.boxes:
#             box = result.boxes[0]
#             class_id = int(box.cls)
#             object_name = model.names[class_id]
#             print(f"Detected object: {object_name}")

# # Display the annotated image
# cv.imshow("YOLOv8 Inference", annotated_image)

# # Wait for a key press and close the window
# cv.waitKey(0)
# cv.destroyAllWindows()





# import matplotlib.pyplot as plt
# from ultralytics import YOLO
# import cv2 as cv

# # Load YOLO model
# model = YOLO("best.pt")

# # Load the image you want to detect from
# image_path = "test-2.png"  # Replace with your image path
# image = cv.imread(image_path)

# # Run YOLOv8 inference on the image
# results = model(image, conf=0.1)  # Adjust confidence threshold as needed

# # Annotate the image with bounding boxes and labels
# annotated_image = results[0].plot()  # Use results[0] to get the first result

# # Print detected object names
# if len(results[0]) > 0:
#     for result in results[0]:
#         if result.boxes:
#             box = result.boxes[0]
#             class_id = int(box.cls)
#             object_name = model.names[class_id]
#             print(f"Detected object: {object_name}")

# # Display the annotated image using Matplotlib
# plt.imshow(cv.cvtColor(annotated_image, cv.COLOR_BGR2RGB))  # Convert BGR to RGB for Matplotlib
# plt.axis('off')  # Turn off axis
# plt.show()
