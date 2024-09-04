import os
import datetime
import cv2
from ultralytics import YOLOWorld
from ultralytics.utils.plotting import Annotator, colors
import output_tools
import streamlit as st
import tempfile
    
# Set up the Streamlit interface
st.set_page_config(
    page_title="AutoLabel",
    page_icon=":tada:",
    layout="wide",
)

# Page title name
st.title('Real-time Video AutoLabel with Streamlit and YoloWorld')

# Create a two-column layout
col1, col2 = st.columns([1, 2])

# Initialize default values
default_prompt = "person, dog"
default_weight = "/path/to/yourWeight.pt"
default_threshold = "0.1"
# default_output = ""
video_source = None
output_folder = None
default_train_ratio = "0.7"
default_val_ratio = "0.2"
default_test_ratio = "0.1"

with col1:
    # Create file uploader
    uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov'])
    
    # Initialize session state for weight if not already done
    if 'weight_value' not in st.session_state:
        st.session_state.weight_value = default_weight
    
    # Create text input fields for pre-train weight
    weight = st.text_input("Weight", value=st.session_state.weight_value)

    # Create column buttons layout for fast changing pre-train weight
    buttonS, buttonM, buttonL, buttonX = st.columns(4)

    # Place buttons in each column
    with buttonS:
        if st.button("yolov8s-worldv2"):
            st.session_state.weight_value = "./pretrain_weight/yolov8s-worldv2.pt"
    with buttonM:
        if st.button("yolov8m-worldv2"):
            st.session_state.weight_value = "./pretrain_weight/yolov8m-worldv2.pt"
    with buttonL:
        if st.button("yolov8l-worldv2"):
            st.session_state.weight_value = "./pretrain_weight/yolov8l-worldv2.pt"
    with buttonX:
        if st.button("yolov8x-worldv2"):
            st.session_state.weight_value = "./pretrain_weight/yolov8x-worldv2.pt"


    # Create text input fields for prompt, threshold and output folder
    prompt = st.text_input("Prompt", default_prompt)
    threshold = st.text_input("Threshold", default_threshold)
    output_folder = st.text_input("Output Folder Name")

    # Create column buttons layout for fast changing pre-train weight
    train_ratio, val_ratio, test_ratio = st.columns(3)
    # Place buttons in each column
    with train_ratio:
        train_data_ratio = st.text_input("train_ratio", default_train_ratio)
    with val_ratio:
        val_data_ratio = st.text_input("val_ratio", default_val_ratio)
    with test_ratio:
        test_data_ratio = st.text_input("test_ratio", default_test_ratio)

    # Create a submit button for upload video
    if st.button("Submit"):
        old_folder = os.listdir('./datasets/')
        if uploaded_file is None:
            st.warning("Please upload a video file.")
        elif output_folder == '' or output_folder in old_folder:
            st.warning("Please enter a unique folder path.")
        elif not (0 < float(train_data_ratio) < 1) or not (0 < float(val_data_ratio) < 1) or (0 > float(test_data_ratio) or float(test_data_ratio) > 1):
            st.warning("Ratio values must be between 0 and 1.")
        elif not (abs((float(train_data_ratio) + float(val_data_ratio) + float(test_data_ratio)) - 1) < 1e-6):
            st.warning("The sum of train_ratio, valid_ratio, and test_ratio must be 1.")
        else:
            # Save the uploaded video to a temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            with open(temp_file.name, 'wb') as f:
                f.write(uploaded_file.read())
            
            video_source = temp_file.name
            
            prompt_list = [item.strip() for item in prompt.split(',')]
            
            # Show the indo of user chosen parameter
            # st.write(f"Weight: {weight}")
            # st.write(f"Prompt: {prompt_list}")
            # st.write(f"Threshold: {threshold}")

            # Update `run_image` to True to start processing
            st.session_state.run_image = True

# Ensure `run_image` exists in session state
if 'run_image' not in st.session_state:
    st.session_state.run_image = False

with col2:
    # Create a placeholder for changing frame and status messages
    frame_placeholder = st.empty()
    status_placeholder = st.empty() 
    
    # Setup for real-time stream width
    new_width = 480
    status_placeholder.info('Wait for press the submit button...')
    if st.session_state.run_image:

        # Get current date and time
        now = datetime.datetime.now()
        
        # formated Date and Time (YYYY-MM-DD_HH-MM-SS)
        fileDateTime = now.strftime("%Y_%m_%d_%H_%M_%S")

        # Creating Output Folder
        total_folder_path = "datasets/" + output_folder
        total_folder_path = os.path.join('./', total_folder_path)
        detection_yolo = total_folder_path + "/Roboflow_YOLO"
        if not os.path.exists(total_folder_path):
            output_tools.gen_only_yolo_folder(total_folder_path,detection_yolo)
        
        # YoloWorld Parameters
        confidence = float(threshold) if threshold else 0.5
        image_size = 640  
        half_precision = False

        # Load a pretrained YOLOv8s-worldv2 model
        yoloworld_model = YOLOWorld(str(weight))
        yoloworld_model.set_classes(prompt_list)
        yoloworld_classname = yoloworld_model.names

        # Open video capture
        cap = cv2.VideoCapture(video_source)

        # For store image & txt filename
        count = 0  

        # Get total number of frames in the video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Total frames in the video: {total_frames}")

        # Prepare video writer
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(total_folder_path + '/output.mp4', fourcc, fps, (width, height))
        status_placeholder.warning('Running AutoLabel Program!! Please do not close the app!!')

        # Loop through the video frames
        while cap.isOpened():
            # Read a frame from the video
            success, frame = cap.read()

            if success:
                annotated_frame = frame.copy()
                
                # Storing pictures
                cv2.imwrite(detection_yolo+'/datasets/images/'+fileDateTime+str(count)+'.jpg', frame)
                print("======================")
                print("Object Detection:")
                
                # Using frame to predict object 
                predictions = yoloworld_model(frame, save_txt=None, conf=confidence, imgsz=image_size, half=half_precision)

                # Storing labels
                with open(detection_yolo+'/datasets/labels/'+ fileDateTime +str(count)+".txt", 'w') as file:
                    # change final attribute to desired box format
                    for idx, prediction in enumerate(predictions[0].boxes.xywhn): 
                        cls = int(predictions[0].boxes.cls[idx].item())
                        # Write line to file in YOLO label format : cls x y w h
                        file.write(f"{cls} {prediction[0].item()} {prediction[1].item()} {prediction[2].item()} {prediction[3].item()}\n")

                boxes = predictions[0].boxes.xyxy.cpu().tolist()
                clss = predictions[0].boxes.cls.cpu().tolist()
                confidences = predictions[0].boxes.conf.cpu().tolist()
                annotator = Annotator(frame, font_size=3, line_width=4, example=yoloworld_classname)
                if boxes:
                    for box, cls, conf in zip(boxes, clss, confidences):
                        annotator.box_label(box, color=colors(int(cls), True), label=yoloworld_classname[int(cls)])
                annotated_frame = annotator.result()
                out.write(annotated_frame)

                # Display the frame with bounding boxes in Streamlit
                resized_frame = output_tools.resize_frame(annotated_frame, 480)
                base64_image = output_tools.encode_image_to_base64(resized_frame)
                
                # Add CSS to center the image
                frame_placeholder.markdown(
                    f"""
                    <style>
                    .centered {{
                        display: flex;
                        justify-content: center;
                    }}
                    </style>
                    <div class="centered">
                        <img src="data:image/jpeg;base64,{base64_image}" />
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                count += 1
            else:
                break

        cap.release()
        out.release()

        # # Delete the temporary video file
        # os.remove(video_source)

        status_placeholder.success('AutoLabel Program Finished!')
        print("\ncreating file of Roboflow:\n")
        output_tools.genAllDatayaml(detection_yolo, prompt_list)
        output_tools.split_dataset(detection_yolo,train_data_ratio,val_data_ratio)
    st.session_state.run_image = False
