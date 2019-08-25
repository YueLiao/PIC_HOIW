from visulize_hoiw import visualize
import argparse
parser = argparse.ArgumentParser(description='HOI demo')
parser.add_argument('--annot_file', type=str, help='annotation file path')
parser.add_argument('--image_dir', type=str, help='image dictionary')
parser.add_argument('--image_name', default='', type=str,
                    help='image file')
parser.add_argument('--output_path', default='', type=str,
                    help='vis result path')
args = parser.parse_args()

hoi_vis = visualize(args.image_dir)
hoi_vis.load_annot(args.annot_file)

if len(args.image_name) == 0 and len(args.output_path) > 0:
    hoi_vis.vis_all_images(args.output_path)
elif len(args.image_name) > 0:
    hoi_vis.vis_one_image(args.image_name)

