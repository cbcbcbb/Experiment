import os 
import time
import argparse
from propainter.inference import Propainter, get_device

def main():

    ## input params
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_video', type=str, default="examples/example6/video.mp4", help='Path to the input video')
    parser.add_argument('--input_mask', type=str, default="examples/example6/mask.mp4" , help='Path to the input mask')
    parser.add_argument('--video_length', type=int, default=10, help='The maximum length of output video')
    parser.add_argument('--mask_dilation_iter', type=int, default=8, help='Adjust it to change the degree of mask expansion')
    parser.add_argument('--save_path', type=str, default="results" , help='Path to the output')
    parser.add_argument('--ref_stride', type=int, default=10, help='Propainter params')
    parser.add_argument('--neighbor_length', type=int, default=10, help='Propainter params')
    parser.add_argument('--subvideo_length', type=int, default=50, help='Propainter params')
    parser.add_argument('--propainter_model_dir', type=str, default="weights/propainter" , help='Path to priori model')
    args = parser.parse_args()
                  
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    priori_path = os.path.join(args.save_path, "priori.mp4")                        
    
    ## model initialization
    device = get_device()

    propainter = Propainter(args.propainter_model_dir, device=device)
    
    start_time = time.time()

    ## priori
    propainter.forward(args.input_video, args.input_mask, priori_path, video_length=args.video_length, 
                        ref_stride=args.ref_stride, neighbor_length=args.neighbor_length, subvideo_length = args.subvideo_length,
                        mask_dilation = args.mask_dilation_iter) 


if __name__ == '__main__':
    main()


   