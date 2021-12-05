#%% Constants, imports, environment

import argparse
import json
import os
import sys
import time
import copy
import warnings
import itertools

from datetime import datetime
from time import sleep
from functools import partial
from tempfile import NamedTemporaryFile

import humanfriendly
from tqdm import tqdm
# from multiprocessing.pool import ThreadPool as workerpool
from multiprocessing.pool import Pool as workerpool
from ct_utils import truncate_float
import numpy as np
import glob


# flasd = open('progres.txt', 'w')


# f = open(f.name, 'w')

# Useful hack to force CPU inference
#
# Need to do this before any TF imports
with open('progress.txt', 'w') as flasd:
    flasd.write('0\n0')

force_cpu = False
if force_cpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class ImagePathUtils:
    """A collection of utility functions supporting this stand-alone script"""

    # Stick this into filenames before the extension for the rendered result
    DETECTION_FILENAME_INSERT = '_detections'

    image_extensions = ['.jpg', '.jpeg', '.png']

    @staticmethod
    def is_image_file(s):
        """
        Check a file's extension against a hard-coded set of image file extensions
        """
        ext = os.path.splitext(s)[1]
        return ext.lower() in ImagePathUtils.image_extensions

    @staticmethod
    def find_image_files(strings):
        """
        Given a list of strings that are potentially image file names, look for strings
        that actually look like image file names (based on extension).
        """
        return [s for s in strings if ImagePathUtils.is_image_file(s)]

    @staticmethod
    def find_images(dir_name, recursive=False):
        """
        Find all files in a directory that look like image file names
        """
        if recursive:
            strings = glob.glob(os.path.join(dir_name, '**', '*.*'), recursive=True)
        else:
            strings = glob.glob(os.path.join(dir_name, '*.*'))

        image_strings = ImagePathUtils.find_image_files(strings)

        return image_strings


class TFDetector:
    """
    A detector model loaded at the time of initialization. It is intended to be used with
    the MegaDetector (TF). The inference batch size is set to 1; code needs to be modified
    to support larger batch sizes, including resizing appropriately.
    """

    # Number of decimal places to round to for confidence and bbox coordinates
    CONF_DIGITS = 3
    COORD_DIGITS = 4

    # MegaDetector was trained with batch size of 1, and the resizing function is a part
    # of the inference graph
    BATCH_SIZE = 1

    # An enumeration of failure reasons
    FAILURE_TF_INFER = 'Failure TF inference'
    FAILURE_IMAGE_OPEN = 'Failure image access'

    DEFAULT_RENDERING_CONFIDENCE_THRESHOLD = 0.85  # to render bounding boxes
    DEFAULT_OUTPUT_CONFIDENCE_THRESHOLD = 0.1  # to include in the output json file

    DEFAULT_DETECTOR_LABEL_MAP = {
        '1': 'animal',
        '2': 'person'
        # '3': 'vehicle'  # available in megadetector v4+
    }

    NUM_DETECTOR_CATEGORIES = 4  # animal, person, group, vehicle - for color assignment

    def __init__(self, model_path):
        """Loads model from model_path and starts a tf.Session with this graph. Obtains
        input and output tensor handles."""
        detection_graph = TFDetector.__load_model(model_path)
        self.tf_session = tf.compat.v1.Session(graph=detection_graph)

        self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        self.box_tensor = detection_graph.get_tensor_by_name('detection_boxes:0')
        self.score_tensor = detection_graph.get_tensor_by_name('detection_scores:0')
        self.class_tensor = detection_graph.get_tensor_by_name('detection_classes:0')

    @staticmethod
    def round_and_make_float(d, precision=4):
        return truncate_float(float(d), precision=precision)

    @staticmethod
    def __convert_coords(tf_coords):
        """Converts coordinates from the model's output format [y1, x1, y2, x2] to the
        format used by our API and MegaDB: [x1, y1, width, height]. All coordinates
        (including model outputs) are normalized in the range [0, 1].
        Args:
            tf_coords: np.array of predicted bounding box coordinates from the TF detector,
                has format [y1, x1, y2, x2]
        Returns: list of Python float, predicted bounding box coordinates [x1, y1, width, height]
        """
        # change from [y1, x1, y2, x2] to [x1, y1, width, height]
        width = tf_coords[3] - tf_coords[1]
        height = tf_coords[2] - tf_coords[0]

        new = [tf_coords[1], tf_coords[0], width, height]  # must be a list instead of np.array

        # convert numpy floats to Python floats
        for i, d in enumerate(new):
            new[i] = TFDetector.round_and_make_float(d, precision=TFDetector.COORD_DIGITS)
        return new

    @staticmethod
    def convert_to_tf_coords(array):
        """From [x1, y1, width, height] to [y1, x1, y2, x2], where x1 is x_min, x2 is x_max
        This is an extraneous step as the model outputs [y1, x1, y2, x2] but were converted to the API
        output format - only to keep the interface of the sync API.
        """
        x1 = array[0]
        y1 = array[1]
        width = array[2]
        height = array[3]
        x2 = x1 + width
        y2 = y1 + height
        return [y1, x1, y2, x2]

    @staticmethod
    def __load_model(model_path):
        """Loads a detection model (i.e., create a graph) from a .pb file.
        Args:
            model_path: .pb file of the model.
        Returns: the loaded graph.
        """
        print('TFDetector: Loading graph...')
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.compat.v2.io.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        print('TFDetector: Detection graph loaded.')

        return detection_graph

    def _generate_detections_one_image(self, image):
        np_im = np.asarray(image, np.uint8)
        im_w_batch_dim = np.expand_dims(np_im, axis=0)

        # need to change the above line to the following if supporting a batch size > 1 and resizing to the same size
        # np_images = [np.asarray(image, np.uint8) for image in images]
        # images_stacked = np.stack(np_images, axis=0) if len(images) > 1 else np.expand_dims(np_images[0], axis=0)

        # performs inference
        (box_tensor_out, score_tensor_out, class_tensor_out) = self.tf_session.run(
            [self.box_tensor, self.score_tensor, self.class_tensor],
            feed_dict={self.image_tensor: im_w_batch_dim})

        return box_tensor_out, score_tensor_out, class_tensor_out

    def generate_detections_one_image(self, image, image_id,
                                      detection_threshold=DEFAULT_OUTPUT_CONFIDENCE_THRESHOLD):
        """Apply the detector to an image.
        Args:
            image: the PIL Image object
            image_id: a path to identify the image; will be in the "file" field of the output object
            detection_threshold: confidence above which to include the detection proposal
        Returns:
        A dict with the following fields, see the 'images' key in https://github.com/microsoft/CameraTraps/tree/master/api/batch_processing#batch-processing-api-output-format
            - 'file' (always present)
            - 'max_detection_conf'
            - 'detections', which is a list of detection objects containing keys 'category', 'conf' and 'bbox'
            - 'failure'
        """
        result = {
            'file': image_id
        }
        try:
            b_box, b_score, b_class = self._generate_detections_one_image(image)

            # our batch size is 1; need to loop the batch dim if supporting batch size > 1
            boxes, scores, classes = b_box[0], b_score[0], b_class[0]

            detections_cur_image = []  # will be empty for an image with no confident detections
            max_detection_conf = 0.0
            for b, s, c in zip(boxes, scores, classes):
                if s > detection_threshold:
                    detection_entry = {
                        'category': str(int(c)),  # use string type for the numerical class label, not int
                        'conf': truncate_float(float(s),  # cast to float for json serialization
                                               precision=TFDetector.CONF_DIGITS),
                        'bbox': TFDetector.__convert_coords(b)
                    }
                    detections_cur_image.append(detection_entry)
                    if s > max_detection_conf:
                        max_detection_conf = s

            result['max_detection_conf'] = truncate_float(float(max_detection_conf),
                                                          precision=TFDetector.CONF_DIGITS)
            result['detections'] = detections_cur_image

        except Exception as e:
            result['failure'] = TFDetector.FAILURE_TF_INFER
            print('TFDetector: image {} failed during inference: {}'.format(image_id, str(e)))

        return result

import visualization_utils as viz_utils

# Numpy FutureWarnings from tensorflow import
warnings.filterwarnings('ignore', category=FutureWarning)

import tensorflow as tf

# print('TensorFlow version:', tf.__version__)
# print('tf.test.is_gpu_available:', tf.test.is_gpu_available())


#%% Support functions for multiprocessing

def process_images(im_files, tf_detector, confidence_threshold):
    """Runs the MegaDetector over a list of image files.

    Args
    - im_files: list of str, paths to image files
    - tf_detector: TFDetector (loaded model) or str (path to .pb model file)
    - confidence_threshold: float, only detections above this threshold are returned

    Returns
    - results: list of dict, each dict represents detections on one image
        see the 'images' key in https://github.com/microsoft/CameraTraps/tree/master/api/batch_processing#batch-processing-api-output-format
    """
    if isinstance(tf_detector, str):
        start_time = time.time()
        tf_detector = TFDetector(tf_detector)
        elapsed = time.time() - start_time
        print('Loaded model (batch level) in {}'.format(humanfriendly.format_timespan(elapsed)))

    results = []
    for im_file in im_files:
        results.append(process_image(im_file, tf_detector, confidence_threshold))
    return results


def process_image(im_file, tf_detector, confidence_threshold):
    """Runs the MegaDetector over a single image file.

    Args
    - im_file: str, path to image file
    - tf_detector: TFDetector, loaded model
    - confidence_threshold: float, only detections above this threshold are returned

    Returns:
    - result: dict representing detections on one image
        see the 'images' key in https://github.com/microsoft/CameraTraps/tree/master/api/batch_processing#batch-processing-api-output-format
    """
    print('Processing image {}'.format(im_file))
    try:
        image = viz_utils.load_image(im_file)
    except Exception as e:
        print('Image {} cannot be loaded. Exception: {}'.format(im_file, e))
        result = {
            'file': im_file,
            'failure': TFDetector.FAILURE_IMAGE_OPEN
        }
        return result

    try:
        result = tf_detector.generate_detections_one_image(
            image, im_file, detection_threshold=confidence_threshold)
    except Exception as e:
        print('Image {} cannot be processed. Exception: {}'.format(im_file, e))
        result = {
            'file': im_file,
            'failure': TFDetector.FAILURE_TF_INFER
        }
        return result

    return result


def chunks_by_number_of_chunks(ls, n):
    """Splits a list into n even chunks.

    Args
    - ls: list
    - n: int, # of chunks
    """
    for i in range(0, n):
        yield ls[i::n]


#%% Main function

def load_and_run_detector_batch(model_file, image_file_names, destination_path,
                                confidence_threshold=0,
                                results=None, n_cores=0):
    """
    Args
    - model_file: str, path to .pb model file
    - image_file_names: list of str, paths to image files
    - checkpoint_path: str, path to JSON checkpoint file
    - confidence_threshold: float, only detections above this threshold are returned
    - checkpoint_frequency: int, write results to JSON checkpoint file every N images
    - results: list of dict, existing results loaded from checkpoint
    - n_cores: int, # of CPU cores to use

    Returns
    - results: list of dict, each dict represents detections on one image
    """

    if n_cores > 1 and tf.test.is_gpu_available():
        print('Warning: multiple cores requested, but a GPU is available; parallelization across GPUs is not currently supported, defaulting to one GPU')


    # Load the detector
    start_time = time.time()
    tf_detector = TFDetector(model_file)
    elapsed = time.time() - start_time
    print('Loaded model in {}'.format(humanfriendly.format_timespan(elapsed)))

    # Does not count those already processed
    count = 0
    
    qre = tqdm(image_file_names)
    iteration = 0
    last_iteration = len(qre)-1
    for im_file in qre:
        with open('progress.txt', 'w') as flasd:
            flasd.write(str(round(qre.format_dict['n']/qre.format_dict['total']*100))+'\n')
            flasd.write(str(round(qre.format_dict['elapsed'])))
            # print("напечатал")
            # print(round(qre.format_dict['n']/qre.format_dict['total']*100))
            # print(round(qre.format_dict['elapsed']))
            # sleep(60)

        count += 1

        result = process_image(im_file, tf_detector, confidence_threshold)
        # print(result)
        source_path = os.path.abspath(im_file).replace('\\','/')
        if len(result['detections']):
            # print(destination_path)
            for detection in result['detections']:
                if detection['category'] == '1':
                    print(f"Перемещаем в хорошую {im_file}")
                    os.replace(source_path, destination_path + f'/good/{os.path.basename(im_file)}' )
                    break
                elif detection['category'] == '2':
                    print(f"Перемещаем в среднюю {im_file}")
                    os.replace(source_path, destination_path + f'/medium/{os.path.basename(im_file)}')
                    break
        else:
            print(f"Перемещаем в плохую {im_file}")
            os.replace(source_path, destination_path + f'/bad/{os.path.basename(im_file)}')
        # print(len(result['detections']))
        # print(result['detections']['category'])
        
        
        
        if iteration == last_iteration:
            with open('progress.txt', 'w') as flasd:
                flasd.write(str(100)+'\n')
                flasd.write(str(round(qre.format_dict['elapsed'])))
        iteration += 1
            

    # results may have been modified in place, but we also return it for backwards-compatibility.



def write_results_to_file(results, output_file, relative_path_base=None):
    """Writes list of detection results to JSON output file. Format matches
    https://github.com/microsoft/CameraTraps/tree/master/api/batch_processing#batch-processing-api-output-format

    Args
    - results: list of dict, each dict represents detections on one image
    - output_file: str, path to JSON output file, should end in '.json'
    - relative_path_base: str, path to a directory as the base for relative paths
    """
    if relative_path_base is not None:
        results_relative = []
        for r in results:
            r_relative = copy.copy(r)
            r_relative['file'] = os.path.relpath(r_relative['file'], start=relative_path_base)
            results_relative.append(r_relative)
        results = results_relative

    final_output = {
        'images': results,
        'detection_categories': TFDetector.DEFAULT_DETECTOR_LABEL_MAP,
        'info': {
            'detection_completion_time': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
            'format_version': '1.0'
        }
    }
    with open(output_file, 'w') as f:
        json.dump(final_output, f, indent=1)
    print('Output file saved at {}'.format(output_file))

def create_directories(directory):
    if not os.path.exists(directory+"/good"):
        os.mkdir(directory+"/good")
    if not os.path.exists(directory+"/medium"):
        os.mkdir(directory+"/medium")
    if not os.path.exists(directory+"/bad"):
        os.mkdir(directory+"/bad")


#%% Command-line driver

def main():
    
    parser = argparse.ArgumentParser(
        description='Module to run a TF animal detection model on lots of images')
    parser.add_argument(
        'detector_file',
        help='Path to .pb TensorFlow detector model file')
    parser.add_argument(
        'image_file',
        help='Path to a single image file, a JSON file containing a list of paths to images, or a directory')
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Recurse into directories, only meaningful if image_file points to a directory')
    parser.add_argument(
        '--output_relative_filenames',
        action='store_true',
        help='Output relative file names, only meaningful if image_file points to a directory')
    parser.add_argument(
        '--threshold',
        type=float,
        default=TFDetector.DEFAULT_OUTPUT_CONFIDENCE_THRESHOLD,
        help="Confidence threshold between 0 and 1.0, don't include boxes below this confidence in the output file. Default is 0.1")
    parser.add_argument(
        '--checkpoint_frequency',
        type=int,
        default=-1,
        help='Write results to a temporary file every N images; default is -1, which disables this feature')
    parser.add_argument(
        '--resume_from_checkpoint',
        help='Path to a JSON checkpoint file to resume from, must be in same directory as output_file')
    parser.add_argument(
        '--ncores',
        type=int,
        default=0,
        help='Number of cores to use; only applies to CPU-based inference, does not support checkpointing when ncores > 1')

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()

    assert os.path.exists(args.detector_file), 'Specified detector_file does not exist'
    assert 0.0 < args.threshold <= 1.0, 'Confidence threshold needs to be between 0 and 1'  # Python chained comparison
    # assert args.output_file.endswith('.json'), 'output_file specified needs to end with .json'
    if args.checkpoint_frequency != -1:
        assert args.checkpoint_frequency > 0, 'Checkpoint_frequency needs to be > 0 or == -1'
    if args.output_relative_filenames:
        assert os.path.isdir(args.image_file), 'image_file must be a directory when --output_relative_filenames is set'

    create_directories(args.image_file)


    # Load the checkpoint if available
    #
    # Relative file names are only output at the end; all file paths in the checkpoint are
    # still full paths.
    if args.resume_from_checkpoint:
        assert os.path.exists(args.resume_from_checkpoint), 'File at resume_from_checkpoint specified does not exist'
        with open(args.resume_from_checkpoint) as f:
            saved = json.load(f)
        assert 'images' in saved, \
            'The file saved as checkpoint does not have the correct fields; cannot be restored'
        results = saved['images']
        print('Restored {} entries from the checkpoint'.format(len(results)))
    else:
        results = []

    # Find the images to score; images can be a directory, may need to recurse
    if os.path.isdir(args.image_file):
        image_file_names = ImagePathUtils.find_images(args.image_file, args.recursive)
        print('{} image files found in the input directory'.format(len(image_file_names)))
    # A json list of image paths
    elif os.path.isfile(args.image_file) and args.image_file.endswith('.json'):
        with open(args.image_file) as f:
            image_file_names = json.load(f)
        print('{} image files found in the json list'.format(len(image_file_names)))
    # A single image file
    elif os.path.isfile(args.image_file) and ImagePathUtils.is_image_file(args.image_file):
        image_file_names = [args.image_file]
        print('A single image at {} is the input file'.format(args.image_file))
    else:
        raise ValueError('image_file specified is not a directory, a json list, or an image file, '
                         '(or does not have recognizable extensions).')

    assert len(image_file_names) > 0, 'Specified image_file does not point to valid image files'
    assert os.path.exists(image_file_names[0]), 'The first image to be scored does not exist at {}'.format(image_file_names[0])



    # Test that we can write to the output_file's dir if checkpointing requested
    # if args.checkpoint_frequency != -1:
    #     checkpoint_path = os.path.join(output_dir, 'checkpoint_{}.json'.format(datetime.utcnow().strftime("%Y%m%d%H%M%S")))
    #     with open(checkpoint_path, 'w') as f:
    #         json.dump({'images': []}, f)
    #     print('The checkpoint file will be written to {}'.format(checkpoint_path))
    # else:
    #     checkpoint_path = None

    start_time = time.time()

    load_and_run_detector_batch(model_file=args.detector_file,
                                image_file_names=image_file_names,
                                destination_path=args.image_file,
                                confidence_threshold=args.threshold,
                                results=results,
                                n_cores=args.ncores)

    elapsed = time.time() - start_time
    print('Finished inference in {}'.format(humanfriendly.format_timespan(elapsed)))

    # relative_path_base = None
    # if args.output_relative_filenames:
    #     relative_path_base = args.image_file
    # write_results_to_file(results, args.output_file, relative_path_base=relative_path_base)

    # if checkpoint_path:
    #     os.remove(checkpoint_path)
    #     print('Deleted checkpoint file')

    print('Done!')


if __name__ == '__main__':
    main()