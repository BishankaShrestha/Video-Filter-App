"""
VideoVision: Smart Video Frame Filtering App
Main Streamlit application for video frame filtering and processing.
"""

import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from PIL import Image
import io
import time

# Import our custom modules
from utils.frame_utils import (
    extract_frame, get_video_info, bgr_to_rgb, 
    resize_frame, save_frame_as_temp_file, validate_video_file
)
from filters.basic_filters import (
    apply_grayscale, apply_blur, apply_sharpen,
    apply_brightness, apply_contrast
)
from filters.edge_filters import (
    apply_canny, apply_sobel, apply_laplacian,
    apply_prewitt
)

# Page configuration
st.set_page_config(
    page_title="VideoVision: Smart Video Frame Filtering App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .filter-preview {
        border: 2px solid #ddd;
        border-radius: 8px;
        padding: 10px;
        margin: 5px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .filter-preview:hover {
        border-color: #1f77b4;
        transform: scale(1.05);
    }
    .filter-preview.selected {
        border-color: #1f77b4;
        background-color: #e6f3ff;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 1rem;
        color: #666;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_frame' not in st.session_state:
    st.session_state.current_frame = None
if 'selected_filter' not in st.session_state:
    st.session_state.selected_filter = None
if 'filter_params' not in st.session_state:
    st.session_state.filter_params = {}
if 'video_info' not in st.session_state:
    st.session_state.video_info = None
if 'temp_video_path' not in st.session_state:
    st.session_state.temp_video_path = None
if 'is_playing' not in st.session_state:
    st.session_state.is_playing = False
if 'current_frame_number' not in st.session_state:
    st.session_state.current_frame_number = 0
if 'last_play_time' not in st.session_state:
    st.session_state.last_play_time = 0

def apply_filter(frame, filter_name, params):
    """Apply the selected filter to the frame."""
    if frame is None or filter_name is None:
        return frame
    
    try:
        if filter_name == 'grayscale':
            return apply_grayscale(frame)
        elif filter_name == 'blur':
            kernel_size = params.get('kernel_size', 5)
            return apply_blur(frame, kernel_size)
        elif filter_name == 'sharpen':
            intensity = params.get('intensity', 1.0)
            return apply_sharpen(frame, intensity)
        elif filter_name == 'canny':
            threshold1 = params.get('threshold1', 50)
            threshold2 = params.get('threshold2', 150)
            return apply_canny(frame, threshold1, threshold2)
        elif filter_name == 'sobel':
            direction = params.get('direction', 'both')
            return apply_sobel(frame, direction)
        elif filter_name == 'brightness':
            brightness = params.get('brightness', 0)
            return apply_brightness(frame, brightness)
        elif filter_name == 'contrast':
            contrast = params.get('contrast', 1.0)
            return apply_contrast(frame, contrast)
        elif filter_name == 'laplacian':
            return apply_laplacian(frame)
        elif filter_name == 'prewitt':
            return apply_prewitt(frame)
        elif filter_name == 'hough_transform':
            threshold = params.get('threshold', 100)
            min_line_length = params.get('min_line_length', 50)
            max_line_gap = params.get('max_line_gap', 10)
            return apply_hough_transform(frame, threshold, min_line_length, max_line_gap)
        elif filter_name == 'dilation':
            kernel_size = params.get('kernel_size', 5)
            iterations = params.get('iterations', 1)
            return apply_dilation(frame, kernel_size, iterations)
        elif filter_name == 'erosion':
            kernel_size = params.get('kernel_size', 5)
            iterations = params.get('iterations', 1)
            return apply_erosion(frame, kernel_size, iterations)
        else:
            return frame
    except Exception as e:
        st.error(f"Error applying filter: {e}")
        return frame

def create_filter_preview(frame, filter_name, size=(100, 100)):
    """Create a small preview of the filter effect."""
    if frame is None:
        return None
    
    # Resize frame for preview
    preview_frame = cv2.resize(frame, size)
    
    # Apply filter
    filtered_preview = apply_filter(preview_frame, filter_name, st.session_state.filter_params.get(filter_name, {}))
    
    if filtered_preview is not None:
        # Convert to RGB for display
        rgb_preview = bgr_to_rgb(filtered_preview)
        return rgb_preview
    return None

def get_current_frame_from_video():
    """Get the current frame from the video."""
    if st.session_state.temp_video_path is None:
        return None
    
    try:
        frame = extract_frame(st.session_state.temp_video_path, st.session_state.current_frame_number)
        return frame
    except Exception as e:
        st.error(f"Error getting frame: {e}")
        return None

def play_video():
    """Play video by advancing frames."""
    if not st.session_state.is_playing:
        return
    
    video_info = st.session_state.video_info
    if st.session_state.current_frame_number < video_info['total_frames'] - 1:
        st.session_state.current_frame_number += 1
    else:
        st.session_state.is_playing = False  # End of video

def build_point_cloud_from_image(image_bgr, height_scale=1.0, step=4):
    """Create an Open3D point cloud from a BGR image using intensity as height.

    Args:
        image_bgr: HxWx3 uint8 OpenCV image in BGR format.
        height_scale: Multiplier for the Z (height) derived from intensity.
        step: Subsampling step to reduce number of points.

    Returns:
        o3d.geometry.PointCloud instance.
    """
    if image_bgr is None:
        raise ValueError("Input image is None")

    if step < 1:
        step = 1

    height, width = image_bgr.shape[:2]

    # Colors from RGB, normalized to [0,1]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    colors = image_rgb[0:height:step, 0:width:step].astype(np.float32) / 255.0

    # Height (Z) from grayscale intensity centered around 0
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray_sub = gray[0:height:step, 0:width:step].astype(np.float32) / 255.0
    z = (gray_sub - 0.5) * 2.0 * float(height_scale)

    # XY grid in image coordinates, scaled to [-1, 1]
    ys, xs = np.mgrid[0:height:step, 0:width:step].astype(np.float32)
    xs_norm = (xs / max(width - 1, 1)) * 2.0 - 1.0
    ys_norm = (ys / max(height - 1, 1)) * 2.0 - 1.0

    points = np.stack([xs_norm, -ys_norm, z], axis=-1).reshape(-1, 3)
    colors_flat = colors.reshape(-1, 3)

    return points, colors_flat

def compute_motion_heatmap(current_bgr, previous_bgr, threshold=20, blur_kernel=5):
    """Compute motion heatmap from two consecutive BGR frames.

    Returns a BGR heatmap image where high motion regions are highlighted.
    """
    if current_bgr is None or previous_bgr is None:
        raise ValueError("Both current and previous frames are required")
    if blur_kernel % 2 == 0:
        blur_kernel += 1
    current_gray = cv2.cvtColor(current_bgr, cv2.COLOR_BGR2GRAY)
    previous_gray = cv2.cvtColor(previous_bgr, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(current_gray, previous_gray)
    if blur_kernel > 1:
        diff = cv2.GaussianBlur(diff, (blur_kernel, blur_kernel), 0)
    _, motion_mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_TOZERO)
    norm = cv2.normalize(motion_mask, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_JET)
    return heatmap

def main():
    # Main header
    st.markdown('<h1 class="main-header">🎬 VideoVision: Smart Video Frame Filtering App</h1>', unsafe_allow_html=True)
    
    # Sidebar for controls
    with st.sidebar:
        st.header("Controls")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload a video file",
            type=['mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm'],
            help="Supported formats: MP4, AVI, MOV, MKV, WMV, FLV, WEBM"
        )
        
        if uploaded_file is not None:
            # Validate video file
            if not validate_video_file(uploaded_file):
                st.error("Please upload a valid video file.")
                return
            
            # Save uploaded file temporarily
            if st.session_state.temp_video_path is None or not os.path.exists(st.session_state.temp_video_path):
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    st.session_state.temp_video_path = tmp_file.name
            
            # Get video info
            if st.session_state.video_info is None:
                total_frames, width, height, fps = get_video_info(st.session_state.temp_video_path)
                st.session_state.video_info = {
                    'total_frames': total_frames,
                    'width': width,
                    'height': height,
                    'fps': fps
                }
            
            video_info = st.session_state.video_info
            
            if video_info['total_frames'] > 0:
                st.success(f" Video loaded successfully!")
                st.info(f"Video Info:\n- Frames: {video_info['total_frames']}\n- Resolution: {video_info['width']}x{video_info['height']}\n- FPS: {video_info['fps']:.2f}")
                
                # Video Player Section
                st.subheader(" Video Player")
                
                col1, col2, col3, col4 = st.columns(4)
                
            # """
            # Commented out video player as some latency issue is there.
            #     with col1:
            #         if st.button("▶️ Play", disabled=st.session_state.is_playing):
            #             st.session_state.is_playing = True
            #             st.session_state.last_play_time = time.time()
            #             st.rerun()
                
            #     with col2:
            #         if st.button("⏸️ Pause", disabled=not st.session_state.is_playing):
            #             st.session_state.is_playing = False
            #             st.rerun()
                
            #     with col3:
            #         if st.button("⏹️ Stop"):
            #             st.session_state.is_playing = False
            #             st.session_state.current_frame_number = 0
            #             st.rerun()
            #     """
                
                with col1:
                    if st.button("Extract Current Frame", type="primary"):
                        frame = get_current_frame_from_video()
                        if frame is not None:
                            st.session_state.current_frame = frame
                            st.success(f"Frame {st.session_state.current_frame_number} extracted successfully!")
                        else:
                            st.error("Failed to extract frame.")
                
                # Manual step button for debugging
                if st.button("Step Forward"):
                    if st.session_state.current_frame_number < video_info['total_frames'] - 1:
                        st.session_state.current_frame_number += 1
                        st.rerun()
                    else:
                        st.info("End of video reached")
                
                # Frame navigation
                st.subheader("Frame Navigation")
                
                # Manual frame selection
                selected_frame = st.slider(
                    "Jump to frame",
                    min_value=0,
                    max_value=video_info['total_frames'] - 1,
                    value=st.session_state.current_frame_number,
                    help="Use the slider to navigate through video frames"
                )
                
                # Update current frame number if slider changed
                if selected_frame != st.session_state.current_frame_number:
                    st.session_state.current_frame_number = selected_frame
                    st.session_state.is_playing = False  # Stop playing when manually navigating
                
                # Display current frame info
                st.info(f"Current frame: {st.session_state.current_frame_number} / {video_info['total_frames'] - 1}")
                
                # Auto-play functionality moved to main area
                
                # Remove live preview from sidebar - moved to main area
                
                # Filter selection (only show if frame is extracted)
                if st.session_state.current_frame is not None:
                    st.subheader("Filter Selection")
                    filters = {
                        'grayscale': 'Grayscale',
                        'blur': 'Blur',
                        'sharpen': 'Sharpen',
                        'canny': 'Canny Edge',
                        'sobel': 'Sobel Edge',
                        'brightness': 'Brightness',
                        'contrast': 'Contrast',
                        'laplacian': 'Laplacian',
                        'prewitt': 'Prewitt',
                        'hough_transform': 'Hough Transform',
                        'dilation': 'Dilation',
                        'erosion': 'Erosion'
                    }
                    
                    # Filter selection dropdown
                    st.write("Select Filter:")
                    selected_filter = st.selectbox(
                        "Choose a filter to apply:",
                        options=list(filters.keys()),
                        format_func=lambda x: filters[x],
                        index=list(filters.keys()).index(st.session_state.selected_filter) if st.session_state.selected_filter in filters else 0,
                        key="filter_selector"
                    )
                    
                    if selected_filter != st.session_state.selected_filter:
                        st.session_state.selected_filter = selected_filter
                        st.rerun()
                    
                    # Filter previews inside an expander
                    with st.expander("Preview Filters (optional)", expanded=False):
                        st.write("Click a button to select a filter.")
                        cols = st.columns(3)
                        for i, (filter_key, filter_name) in enumerate(filters.items()):
                            col_idx = i % 3
                            with cols[col_idx]:
                                preview = create_filter_preview(st.session_state.current_frame, filter_key)
                                if preview is not None:
                                    st.image(preview, caption=filter_name, use_column_width=True)
                                    if st.button(f"Select {filter_name}", key=f"select_{filter_key}"):
                                        st.session_state.selected_filter = filter_key
                                        st.rerun()
                    
                    # Filter parameters
                    if st.session_state.selected_filter:
                        st.subheader(f" {filters[st.session_state.selected_filter]} Parameters")
                        
                        # Initialize filter parameters if not exists
                        if st.session_state.selected_filter not in st.session_state.filter_params:
                            st.session_state.filter_params[st.session_state.selected_filter] = {}
                        
                        params = st.session_state.filter_params[st.session_state.selected_filter]
                        
                        # Parameter controls based on selected filter
                        if st.session_state.selected_filter == 'blur':
                            kernel_size = st.slider("Kernel Size", 3, 15, params.get('kernel_size', 5), 2)
                            params['kernel_size'] = kernel_size
                        elif st.session_state.selected_filter == 'sharpen':
                            intensity = st.slider("Intensity", 0.1, 2.0, params.get('intensity', 1.0), 0.1)
                            params['intensity'] = intensity
                        elif st.session_state.selected_filter == 'canny':
                            threshold1 = st.slider("Threshold 1", 10, 200, params.get('threshold1', 50), 10)
                            threshold2 = st.slider("Threshold 2", 50, 300, params.get('threshold2', 150), 10)
                            params['threshold1'] = threshold1
                            params['threshold2'] = threshold2
                        elif st.session_state.selected_filter == 'sobel':
                            direction = st.selectbox("Direction", ['both', 'x', 'y'], index=['both', 'x', 'y'].index(params.get('direction', 'both')))
                            params['direction'] = direction
                        elif st.session_state.selected_filter == 'brightness':
                            brightness = st.slider("Brightness", -100, 100, params.get('brightness', 0), 5)
                            params['brightness'] = brightness
                        elif st.session_state.selected_filter == 'contrast':
                            contrast = st.slider("Contrast", 0.5, 2.0, params.get('contrast', 1.0), 0.1)
                            params['contrast'] = contrast
                        
                        
                        st.session_state.filter_params[st.session_state.selected_filter] = params
                        
                        # Apply filter button
                        if st.button("Apply Filter", type="primary"):
                            with st.spinner("Applying filter..."):
                                filtered_frame = apply_filter(
                                    st.session_state.current_frame, 
                                    st.session_state.selected_filter, 
                                    params
                                )
                                if filtered_frame is not None:
                                    st.session_state.filtered_frame = filtered_frame
                                    st.success("Filter applied successfully!")
                                else:
                                    st.error("Failed to apply filter.")
            else:
                st.error("Invalid video file or unable to read video information.")
        else:
            st.info("Please upload a video file to get started!")
    
    # Main content area
    if st.session_state.temp_video_path is not None and st.session_state.video_info is not None:
        # Live Video Preview in main area
        st.subheader("Live Video Preview")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            current_preview_frame = get_current_frame_from_video()
            if current_preview_frame is not None:
                preview_rgb = bgr_to_rgb(current_preview_frame)
                # Display original size for proper popup enlargement
                st.image(preview_rgb, caption=f"Frame {st.session_state.current_frame_number}", use_column_width=True)
            else:
                st.info("No frame available for preview")
        # """
        # Commented out video state as some latency issue is there.
        # with col2:
        #     st.info(f"Current Frame:{st.session_state.current_frame_number} / {st.session_state.video_info['total_frames'] - 1}")
        #     if st.session_state.is_playing:
        #         st.success(" Video is playing")
        #         st.write(f"FPS:{st.session_state.video_info['fps']:.2f}")
        #         st.write(f"Frame Duration: {1.0/st.session_state.video_info['fps']:.3f}s")
        #     else:
        #         st.info("⏸ Video is paused")
        # """

        
        # Some bug on auto-play functionality.
        
        # # Auto-play functionality with proper timing
        # if st.session_state.is_playing:
        #     current_time = time.time()
        #     frame_duration = 1.0 / st.session_state.video_info['fps']
            
        #     if current_time - st.session_state.last_play_time >= frame_duration:
        #         play_video()
        #         st.session_state.last_play_time = current_time
        #         st.rerun()
        

        # Motion Heatmap (between consecutive frames)
        #In this code, we are comparing N and N-10 frames to compute motion heatmap to show differences.
        #This version will later be upgraded to choose user to select the frames.
        with st.expander("Motion Heatmap", expanded=False):
            if st.session_state.video_info['total_frames'] < 2:
                st.info("Need at least two frames to compute motion.")
            else:
                threshold = st.slider("Sensitivity (Threshold)", 1, 100, 20)
                blur_ksize = st.slider("Blur Kernel (odd)", 1, 25, 5, step=2)
                overlay_alpha = st.slider("Overlay Alpha", 0.0, 1.0, 0.5, 0.05)

                current_fn = st.session_state.current_frame_number
                if current_fn == 0:
                    st.info("Move to a later frame to compare with the previous one.")
                else:
                    try:
                        prev_bgr = extract_frame(st.session_state.temp_video_path, current_fn - 1)
                        curr_bgr = get_current_frame_from_video()
                        if prev_bgr is None or curr_bgr is None:
                            st.error("Unable to read frames for motion heatmap.")
                        else:
                            heatmap_bgr = compute_motion_heatmap(curr_bgr, prev_bgr, threshold=threshold, blur_kernel=blur_ksize)
                            heatmap_rgb = bgr_to_rgb(heatmap_bgr)
                            st.image(resize_frame(heatmap_rgb, max_width=480, max_height=160), caption="Motion Heatmap", use_column_width=False, width=480)

                            # Optional overlay on current frame
                            overlay = st.checkbox("Show overlay on current frame", value=False)
                            if overlay:
                                curr_rgb = bgr_to_rgb(curr_bgr)
                                # Convert both to float for blending
                                overlay_img = cv2.addWeighted(curr_rgb.astype(np.float32), 1 - overlay_alpha, heatmap_rgb.astype(np.float32), overlay_alpha, 0)
                                overlay_img = np.clip(overlay_img, 0, 255).astype(np.uint8)
                                st.image(resize_frame(overlay_img, max_width=480, max_height=160), caption="Overlay", use_column_width=False, width=480)
                    except Exception as e:
                        st.error(f"Error computing motion heatmap: {e}")
        
        # Show extracted frame processing if available
        if st.session_state.current_frame is not None:
            st.subheader("🖼️ Extracted Frame Processing")
            
            # Display mode selection
            display_mode = st.radio(
                "Display Mode",
                ["Original", "Filtered", "Side by Side"],
                horizontal=True
            )
            
            # Create columns for side-by-side display
            if display_mode == "Side by Side":
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📷 Original Frame")
                    original_rgb = bgr_to_rgb(st.session_state.current_frame)
                    resized_original = resize_frame(original_rgb, max_width=360, max_height=110)
                    st.image(resized_original, use_column_width=False, width=360)
                
                with col2:
                    st.subheader("🎨 Filtered Frame")
                    if hasattr(st.session_state, 'filtered_frame') and st.session_state.filtered_frame is not None:
                        filtered_rgb = bgr_to_rgb(st.session_state.filtered_frame)
                        resized_filtered = resize_frame(filtered_rgb, max_width=360, max_height=110)
                        st.image(resized_filtered, use_column_width=False, width=360)
                    else:
                        st.info("No filtered frame available. Apply a filter first.")
            
            elif display_mode == "Original":
                st.subheader("📷 Original Frame")
                original_rgb = bgr_to_rgb(st.session_state.current_frame)
                resized_original = resize_frame(original_rgb, max_width=480, max_height=160)
                st.image(resized_original, use_column_width=False, width=480)
            
            elif display_mode == "Filtered":
                st.subheader("🎨 Filtered Frame")
                if hasattr(st.session_state, 'filtered_frame') and st.session_state.filtered_frame is not None:
                    filtered_rgb = bgr_to_rgb(st.session_state.filtered_frame)
                    resized_filtered = resize_frame(filtered_rgb, max_width=480, max_height=160)
                    st.image(resized_filtered, use_column_width=False, width=480)
                else:
                    st.info("No filtered frame available. Apply a filter first.")

            # 3D Point Cloud View (Plotly only)
            with st.expander("🟣 3D Point Cloud View", expanded=False):
                source_choice = st.selectbox(
                    "Source Image",
                    ["Original Frame", "Filtered Frame (if available)"]
                )
                step = st.slider("Downsample Step (higher = fewer points)", 1, 20, 4)
                height_scale = st.slider("Height Scale", 0.1, 5.0, 1.0, 0.1)

                if st.button("Show Point Cloud", type="primary"):
                    src_bgr = None
                    if source_choice == "Original Frame":
                        src_bgr = st.session_state.current_frame
                    else:
                        if hasattr(st.session_state, 'filtered_frame') and st.session_state.filtered_frame is not None:
                            src_bgr = st.session_state.filtered_frame
                        else:
                            st.error("Filtered frame not available. Apply a filter or choose Original Frame.")
                    if src_bgr is not None:
                        try:
                            points, colors = build_point_cloud_from_image(src_bgr, height_scale=height_scale, step=step)
                            import plotly.graph_objects as go
                            x, y, z = points[:,0], points[:,1], points[:,2]
                            marker=dict(size=2, color=colors, opacity=0.9)
                            fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=marker)])
                            fig.update_layout(scene=dict(xaxis=dict(title='X'), yaxis=dict(title='Y'), zaxis=dict(title='Z')), margin=dict(l=0, r=0, b=0, t=0))
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Failed to create/show point cloud: {e}")
            
            # Save and download section
            if hasattr(st.session_state, 'filtered_frame') and st.session_state.filtered_frame is not None:
                st.subheader("💾 Save & Download")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("💾 Save Filtered Frame", type="primary"):
                        # Save filtered frame
                        temp_path = save_frame_as_temp_file(st.session_state.filtered_frame, f"filtered_frame_{st.session_state.current_frame_number}")
                        if temp_path:
                            st.success(f"Frame saved as: {os.path.basename(temp_path)}")
                            st.session_state.saved_frame_path = temp_path
                
                with col2:
                    if hasattr(st.session_state, 'saved_frame_path') and st.session_state.saved_frame_path:
                        # Create download button
                        with open(st.session_state.saved_frame_path, "rb") as file:
                            st.download_button(
                                label="⬇️ Download Filtered Frame",
                                data=file.read(),
                                file_name=f"filtered_frame_{st.session_state.current_frame_number}.jpg",
                                mime="image/jpeg"
                            )
    
    else:
        # Welcome message
        st.markdown("""
        <div style="text-align: center; margin-top: 3rem;">
            <h2>Welcome to VideoVision! 🎬</h2>
            <p style="font-size: 1.2rem; color: #666;">
                Upload a video file to start filtering and processing frames.
            </p>
            <p style="color: #888;">
                Supported formats: MP4, AVI, MOV, MKV, WMV, FLV, WEBM
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        Made By: Bishanka and Team
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()