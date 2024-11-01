import gradio as gr
import cv2
import numpy as np

# Function to convert 2x3 affine matrix to 3x3 for matrix multiplication
def to_3x3(affine_matrix):
    return np.vstack([affine_matrix, [0, 0, 1]])

# Function to apply transformations based on user inputs
def apply_transform(image, scale, rotation, translation_x, translation_y, flip_horizontal):

    # Convert the image from PIL format to a NumPy array
    image = np.array(image)
    # Pad the image to avoid boundary issues
    pad_size = min(image.shape[0], image.shape[1]) // 2
    image_new = np.zeros((pad_size*2+image.shape[0], pad_size*2+image.shape[1], 3), dtype=np.uint8) + np.array((255,255,255), dtype=np.uint8).reshape(1,1,3)
    image_new[pad_size:pad_size+image.shape[0], pad_size:pad_size+image.shape[1]] = image
    image = np.array(image_new)
    transformed_image = np.array(image)


    ### FILL: Apply Composition Transform 
    # Note: for scale and rotation, implement them around the center of the image （围绕图像中心进行放缩和旋转）
    h=image.shape[0]
    w=image.shape[1]

    tmp_image=np.array(transformed_image)


    center_x=w//2
    center_y=h//2
    #scale
    for i in range(h):
        for j in range(w):
            pretransformed_i=(i-center_y)/scale+center_y #(1-1/scale)*center_y+(1/scale)*i
            pretransformed_j=(j-center_x)/scale+center_x

            pretransformed_i=int(pretransformed_i)
            pretransformed_j=int(pretransformed_j)
            if pretransformed_i>=0 and pretransformed_i<h and pretransformed_j>=0 and pretransformed_j<w:
                transformed_image[i,j]=image[pretransformed_i,pretransformed_j]
            else:
                transformed_image[i,j]=np.array((255,255,255), dtype=np.uint8).reshape(1,1,3)
    tmp_image=np.array(transformed_image)
    #rotate
    for i in range(h):
        for j in range(w):
            i_=h-i-1
            j_=j

            pretransformed_j=(j_-center_x)*np.cos(np.deg2rad(-rotation))-(i_-center_y)*np.sin(np.deg2rad(-rotation))+center_x
            pretransformed_i=(j_-center_x)*np.sin(np.deg2rad(-rotation))+(i_-center_y)*np.cos(np.deg2rad(-rotation))+center_y
            pretransformed_i=h-pretransformed_i-1
            pretransformed_i=int(pretransformed_i)
            pretransformed_j=int(pretransformed_j)
            if pretransformed_i>=0 and pretransformed_i<h and pretransformed_j>=0 and pretransformed_j<w:
                transformed_image[i,j]=tmp_image[pretransformed_i,pretransformed_j]
            else:
                transformed_image[i,j]=np.array((255,255,255), dtype=np.uint8).reshape(1,1,3)

    tmp_image=np.array(transformed_image)

    #translate
    for i in range(h):
        for j in range(w):
            if i-translation_x>=0 and i-translation_x<h and j-translation_y>=0 and j-translation_y<w:
                transformed_image[i,j]=tmp_image[i-translation_x,j-translation_y]
            else:
                transformed_image[i,j]=np.array((255,255,255), dtype=np.uint8).reshape(1,1,3)



    tmp_image=np.array(transformed_image)
    #flip horizontal
    if flip_horizontal:
        for j in range(w):
            transformed_image[:,j]=tmp_image[:,w-j-1]

    return transformed_image




# Gradio Interface
def interactive_transform():
    with gr.Blocks() as demo:
        gr.Markdown("## Image Transformation Playground")
        
        # Define the layout
        with gr.Row():
            # Left: Image input and sliders
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Image")

                scale = gr.Slider(minimum=0.1, maximum=2.0, step=0.1, value=1.0, label="Scale")
                rotation = gr.Slider(minimum=-180, maximum=180, step=1, value=0, label="Rotation (degrees)")
                translation_x = gr.Slider(minimum=-300, maximum=300, step=10, value=0, label="Translation X")
                translation_y = gr.Slider(minimum=-300, maximum=300, step=10, value=0, label="Translation Y")
                flip_horizontal = gr.Checkbox(label="Flip Horizontal")
            
            # Right: Output image
            image_output = gr.Image(label="Transformed Image")
        
        # Automatically update the output when any slider or checkbox is changed
        inputs = [
            image_input, scale, rotation, 
            translation_x, translation_y, 
            flip_horizontal
        ]

        # Link inputs to the transformation function
        image_input.change(apply_transform, inputs, image_output)
        scale.change(apply_transform, inputs, image_output)
        rotation.change(apply_transform, inputs, image_output)
        translation_x.change(apply_transform, inputs, image_output)
        translation_y.change(apply_transform, inputs, image_output)
        flip_horizontal.change(apply_transform, inputs, image_output)

    return demo

# Launch the Gradio interface
interactive_transform().launch()
