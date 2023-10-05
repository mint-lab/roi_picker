from roi_picker import *

if __name__ == "__main__":
    # Add arguments to the parser
    parser = argparse.ArgumentParser(description='ROI')
    parser.add_argument('--org', type=str, help='Image path to load')
    parser.add_argument('--des', type=str, help='Image path to save')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the values of the arguments
    img_path_load = args.org
    img_path_save = args.des


    pd = PolygonDrawer(img_path_load, img_path_save)
    image = pd.roi_process()
    