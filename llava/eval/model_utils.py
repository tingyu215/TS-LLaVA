import numpy as np
from PIL import Image
from decord import VideoReader, cpu



def load_video(vis_path, n_clips=1, num_frm=100):
    """
    Load video frames from a video file.

    Parameters:
    vis_path (str): Path to the video file.
    n_clips (int): Number of clips to extract from the video. Defaults to 1.
    num_frm (int): Number of frames to extract from each clip. Defaults to 100.

    Returns:
    list: List of PIL.Image.Image objects representing video frames.
    """

    # Load video with VideoReader
    vr = VideoReader(vis_path, ctx=cpu(0))
    total_frame_num = len(vr)

    # Currently, this function supports only 1 clip
    assert n_clips == 1

    # Calculate total number of frames to extract
    total_num_frm = min(total_frame_num, num_frm)
    # Get indices of frames to extract
    frame_idx = get_seq_frames(total_frame_num, total_num_frm)
    # Extract frames as numpy array
    img_array = vr.get_batch(frame_idx).asnumpy()   # T H W C

    #############################################
    # in case you want to use cv2 for video loading, uncomment the following lines
    # cap = cv2.VideoCapture(vis_path)
    # if not cap.isOpened():
    #     raise ValueError("Error: Could not open video file.")
    # total_frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # assert n_clips == 1

    # total_num_frm = min(total_frame_num, num_frm)
    # frame_idx = get_seq_frames(total_frame_num, total_num_frm)
        
    # frames = []
    # for idx in frame_idx:
    #     cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    #     ret, frame = cap.read()
    #     if not ret:
    #         print(f"Warning: Could not retrieve frame at index {idx}")
    #         continue
    #     frames.append(frame)
    # img_array = np.array(frames)
    #############################################


    original_size = (img_array.shape[-2], img_array.shape[-3])  # (width, height)
    original_sizes = (original_size,) * total_num_frm

    clip_imgs = [Image.fromarray(img_array[j]) for j in range(total_num_frm)]
    

    return clip_imgs, original_sizes




def get_seq_frames(total_num_frames, desired_num_frames):
    """
    Calculate the indices of frames to extract from a video.

    Parameters:
    total_num_frames (int): Total number of frames in the video.
    desired_num_frames (int): Desired number of frames to extract.

    Returns:
    list: List of indices of frames to extract.
    """

    # Calculate the size of each segment from which a frame will be extracted
    seg_size = float(total_num_frames - 1) / desired_num_frames

    seq = []
    for i in range(desired_num_frames):
        # Calculate the start and end indices of each segment
        start = int(np.round(seg_size * i))
        end = int(np.round(seg_size * (i + 1)))

        # Append the middle index of the segment to the list
        seq.append((start + end) // 2)

    return seq
