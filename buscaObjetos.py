import cv2
from ultralytics import solutions
from datetime import datetime

#cap = cv2.VideoCapture("/mnt/d/temp/CLANF-MARINHA.mp4") # linux ou wsl (windows subsystem for linux)
#cap = cv2.VideoCapture("d:/temp/DESFILE1-SET2024.mp4") # windows (preocupacao com a barra invertida)
cap = cv2.VideoCapture(0) # webcam (caso tenha uma / use o indice da camera)

assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# classes_to_count = [2] ## apenas carros
# classes_to_count = [0, 2] ## carros e pessoas
classes_to_count = [] ## tudo

# veja na dodumentacao as classes possiveis do modelo
# https://docs.ultralytics.com/datasets/segment/coco/#dataset-yaml
# Classes
#names:
#  0: person         1: bicycle     2: car             3: motorcycle      4: airplane      5: bus   
#  6: train          7: truck       8: boat            9: traffic light  10: fire hydrant  11: stop sign
# 12: parking meter 13: bench      14: bird           15: cat            16: dog           17: horse
# 18: sheep         19: cow        20: elephant       21: bear           22: zebra         23: giraffe
# 24: backpack      25: umbrella   26: handbag        27: tie            28: suitcase      29: frisbee
# 30: skis          31: snowboard  32: sports ball    33: kite           34: baseball bat  35: baseball glove
# 36: skateboard    37: surfboard  38: tennis racket  39: bottle         40: wine glass    41: cup
# 42: fork          43: knife      44: spoon          45: bowl           46: banana        47: apple   
# 48: sandwich      49: orange     50: broccoli       51: carrot         52: hot dog       53: pizza   
# 54: donut         55: cake       56: chair          57: couch          58: potted plant  59: bed
# 60: dining table  61: toilet     62: tv             63: laptop         64: mouse         65: remote
# 66: keyboard      67: cell phone 68: microwave      69: oven           70: toaster       71: sink
# 72: refrigerator  73: book       74: clock          75: vase           76: scissors      77: teddy bear
# 78: hair drier    79: toothbrush


# Define region points
# region_points = [(20, 400), (1080, 400)]  # For line counting
region_points = [(350, 150), (350, 300), (620, 300), (620, 150)]  # For rectangle region counting
# region_points = [(20, 400), (1080, 404), (1080, 360), (20, 360), (20, 400)]  # For polygon region counting



# Video writer
date_string = datetime.now().strftime('%d-%m-%Y-%H-%M-%S')
#video_writer = cv2.VideoWriter(f"/mnt/d/temp/CLANF-MARINHA-{date_string}out.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
video_writer = cv2.VideoWriter(f"d:/temp/DESFILE1-SET2024-{date_string}out.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))


# Init Object Counter
counter = solutions.ObjectCounter(
    show=True,  # Display the output
    region=region_points,  # Pass region points
    model="yolo11x.pt",  # model="yolo11n-obb.pt" for object counting using YOLO11 OBB model.
    # classes=[0, 2],  # If you want to count specific classes i.e person and car with COCO pretrained model.q
    show_in=True,  # Display in counts
    show_out=True,  # Display out counts
    line_width=1,  # Adjust the line width for bounding boxes and text display
)

# Process video
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    im0 = counter.count(im0)
    video_writer.write(im0)

    if cv2.waitKey(5)&0xFF == ord('q'):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()