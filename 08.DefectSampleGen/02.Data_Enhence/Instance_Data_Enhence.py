import cv2
import albumentations as A
import os
import argparse
from datetime import datetime
import glob
import numpy as np
import json
import copy


class BatteryDefectAugmenter:
    def __init__(self):
        self.args = self.parse_arguments()
        self.args.input_path = self.clean_path(self.args.input_path)
        self.args.output_dir = self.clean_path(self.args.output_dir)

        self.aug_samples_config = {
            'rotate': 5,
            'brightness_contrast': 5,
            'vertical_flip': 1,
            'horizontal_flip': 1,
            'gaussian_blur': self.args.samples,
            'random_crop': self.args.samples,
            'gaussian_noise': self.args.samples,
            'salt_pepper_noise': self.args.samples,
            'random_scale': self.args.samples,
            'elastic_transform': self.args.samples
        }
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_root = os.path.join(
            self.args.output_dir,
            f"aug_results_{self.timestamp}"
        )
        self.log_file = os.path.join(self.output_root, "augmentation_log.txt")
        self.safe_makedirs(self.output_root)
        self.log("Battery defect image augmentation started")
        self.log(f"Timestamp: {self.timestamp}")
        self.log(f"Input path: {self.args.input_path}")
        self.log(f"Output root: {self.output_root}")
        self.log(f"Selected augmentations: {', '.join(self.args.augmentations)}")
        self.log(f"Sample configuration: {self.aug_samples_config}")
        self.log(f"Recursive mode: {'Enabled' if self.args.recursive else 'Disabled'}")

        self.keypoint_params = A.KeypointParams(
            format='xy',
            remove_invisible=False
        )
        self.augmentations = self.initialize_augmentations()
        if not self.augmentations:
            raise ValueError("No valid augmentations configured.")

    def clean_path(self, path):
        if not path:
            return path
        path = path.strip()
        path = os.path.normpath(path)
        return path

    def safe_makedirs(self, dir_path):
        try:
            abs_dir = os.path.abspath(dir_path)
            os.makedirs(abs_dir, exist_ok=True)
            self.log(f"Created directory (or exists): {abs_dir}")
        except Exception as e:
            self.log(f"Error creating directory {dir_path}: {str(e)}")
            raise

    def wrap_transform(self, transform, needs_polygons=False):
        if needs_polygons:
            return {
                'needs_polygons': True,
                'transform': A.Compose(
                    [transform],
                    keypoint_params=self.keypoint_params
                )
            }
        return {
            'needs_polygons': False,
            'transform': A.Compose([transform])
        }

    def initialize_augmentations(self):
        augs = {}

        if 'rotate' in self.args.augmentations:
            augs['rotate'] = self.wrap_transform(
                A.Rotate(limit=15, p=1.0),
                needs_polygons=True
            )

        if 'brightness_contrast' in self.args.augmentations:
            augs['brightness_contrast'] = self.wrap_transform(
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.10, 0.15),
                    contrast_limit=(-0.10, 0.15),
                    p=1.0
                ),
                needs_polygons=False
            )

        if 'horizontal_flip' in self.args.augmentations:
            augs['horizontal_flip'] = self.wrap_transform(
                A.HorizontalFlip(p=1.0),
                needs_polygons=True
            )

        if 'vertical_flip' in self.args.augmentations:
            augs['vertical_flip'] = self.wrap_transform(
                A.VerticalFlip(p=1.0),
                needs_polygons=True
            )

        if 'gaussian_blur' in self.args.augmentations:
            augs['gaussian_blur'] = self.wrap_transform(
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                needs_polygons=False
            )

        if 'random_crop' in self.args.augmentations:
            augs['random_crop'] = self.wrap_transform(
                A.RandomResizedCrop(
                    height=224,
                    width=224,
                    scale=(0.7, 1.0),
                    ratio=(0.9, 1.1),
                    p=1.0
                ),
                needs_polygons=True
            )

        if 'gaussian_noise' in self.args.augmentations:
            augs['gaussian_noise'] = self.wrap_transform(
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                needs_polygons=False
            )

        if 'salt_pepper_noise' in self.args.augmentations:
            augs['salt_pepper_noise'] = self.wrap_transform(
                A.ISONoise(intensity=(0.1, 0.5), p=1.0),
                needs_polygons=False
            )

        if 'random_scale' in self.args.augmentations:
            augs['random_scale'] = self.wrap_transform(
                A.RandomScale(scale_limit=(-0.2, 0.2), p=1.0),
                needs_polygons=True
            )

        if 'elastic_transform' in self.args.augmentations:
            augs['elastic_transform'] = self.wrap_transform(
                A.ElasticTransform(
                    alpha=1,
                    sigma=50,
                    alpha_affine=50,
                    p=1.0
                ),
                needs_polygons=True
            )

        return augs

    def log(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_message + '\n')
        except Exception as e:
            print(f"Error writing log: {str(e)}")

    def find_labelme_annotation(self, image_path):
        base = os.path.splitext(image_path)[0]
        candidate = f"{base}.json"
        return candidate if os.path.exists(candidate) else None

    def load_labelme_annotation(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        shapes = []
        polygons = []
        for shape in data.get('shapes', []):
            if shape.get('shape_type', 'polygon') != 'polygon':
                continue
            points = shape.get('points', [])
            if len(points) < 3:
                continue
            polygon = [(float(x), float(y)) for x, y in points]
            shapes.append(copy.deepcopy(shape))
            polygons.append(polygon)

        return data, shapes, polygons

    def sanitize_polygons(self, polygons, shapes, height, width):
        valid_polygons = []
        valid_shapes = []
        max_x = max(width - 1.0, 0.0)
        max_y = max(height - 1.0, 0.0)

        for polygon, shape in zip(polygons, shapes):
            cleaned = []
            for x, y in polygon:
                if not np.isfinite(x) or not np.isfinite(y):
                    continue
                clipped_x = float(min(max(x, 0.0), max_x))
                clipped_y = float(min(max(y, 0.0), max_y))
                cleaned.append((clipped_x, clipped_y))
            if len(cleaned) >= 3:
                valid_polygons.append(cleaned)
                valid_shapes.append(shape)

        return valid_polygons, valid_shapes

    def save_labelme_annotation(
        self,
        template,
        shapes,
        polygons,
        image_size,
        image_filename,
        output_path
    ):
        height, width = image_size
        annotation = copy.deepcopy(template)
        annotation['imagePath'] = image_filename
        annotation['imageHeight'] = int(height)
        annotation['imageWidth'] = int(width)
        annotation['imageData'] = None

        new_shapes = []
        for shape, polygon in zip(shapes, polygons):
            new_shape = copy.deepcopy(shape)
            new_shape['points'] = [
                [round(float(x), 4), round(float(y), 4)]
                for x, y in polygon
            ]
            new_shapes.append(new_shape)
        annotation['shapes'] = new_shapes

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(annotation, f, ensure_ascii=False, indent=2)

    def flatten_polygons(self, polygons):
        flat_points = []
        slices = []
        for idx, polygon in enumerate(polygons):
            start = len(flat_points)
            for x, y in polygon:
                flat_points.append((float(x), float(y)))
            slices.append((idx, start, len(flat_points)))
        return flat_points, slices

    def rebuild_polygons(self, keypoints, slices):
        polygons = []
        for _, start, end in slices:
            pts = keypoints[start:end]
            polygons.append([(float(x), float(y)) for x, y in pts])
        return polygons

    def process_image(self, image_path):
        try:
            image_path = self.clean_path(image_path)

            try:
                image = cv2.imdecode(
                    np.fromfile(image_path, dtype=np.uint8),
                    cv2.IMREAD_COLOR
                )
            except Exception as e:
                self.log(f"Failed to read image with cv2: {str(e)}")
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)

            if image is None:
                self.log(f"Error: Could not read image {image_path}")
                return

            annotation_path = self.find_labelme_annotation(image_path)
            if not annotation_path:
                self.log(f"Warning: No Labelme annotation for {image_path}, skipped.")
                return

            annotation_template, shape_templates, base_polygons = (
                self.load_labelme_annotation(annotation_path)
            )
            if not shape_templates or not base_polygons:
                self.log(f"Warning: No valid polygon shapes in {annotation_path}, skipped.")
                return

            image_basename = os.path.basename(image_path)
            image_name, image_ext = os.path.splitext(image_basename)

            try:
                relative_dir = os.path.relpath(
                    os.path.dirname(image_path),
                    self.args.input_path
                )
                if relative_dir == '.':
                    relative_dir = ''
            except ValueError:
                relative_dir = os.path.basename(os.path.dirname(image_path))

            self.log(
                f"Processing image: "
                f"{os.path.join(relative_dir, image_basename) if relative_dir else image_basename}"
            )

            flat_points, poly_slices = self.flatten_polygons(base_polygons)

            for aug_name, aug_info in self.augmentations.items():
                transform = aug_info['transform']
                needs_polygons = aug_info['needs_polygons']
                current_samples = self.aug_samples_config.get(
                    aug_name,
                    self.args.samples
                )
                if current_samples <= 0:
                    continue

                aug_output_dir = os.path.join(
                    self.output_root,
                    aug_name,
                    relative_dir
                )
                self.safe_makedirs(aug_output_dir)

                for sample_idx in range(current_samples):
                    try:
                        inputs = {'image': image}
                        if needs_polygons:
                            inputs['keypoints'] = flat_points

                        result = transform(**inputs)
                        augmented_image = result['image']

                        if needs_polygons:
                            transformed_keypoints = result.get(
                                'keypoints',
                                flat_points
                            )
                            updated_polygons = self.rebuild_polygons(
                                transformed_keypoints,
                                poly_slices
                            )
                            aligned_shapes = shape_templates
                        else:
                            updated_polygons = base_polygons
                            aligned_shapes = shape_templates

                        polygons_to_save, shapes_to_save = self.sanitize_polygons(
                            updated_polygons,
                            aligned_shapes,
                            augmented_image.shape[0],
                            augmented_image.shape[1]
                        )
                        if not polygons_to_save:
                            self.log(
                                f"Warning: {aug_name} sample {sample_idx + 1} "
                                f"produced no valid polygons, skipped."
                            )
                            continue

                        output_filename = (
                            f"{image_name}_{aug_name}_sample{sample_idx + 1}{image_ext}"
                        )
                        output_path = os.path.join(aug_output_dir, output_filename)
                        output_path = self.clean_path(output_path)

                        try:
                            _, img_encoded = cv2.imencode(
                                image_ext,
                                augmented_image
                            )
                            img_encoded.tofile(output_path)
                        except Exception as e:
                            self.log(f"Failed to save with tofile: {str(e)}")
                            cv2.imwrite(output_path, augmented_image)

                        json_filename = f"{image_name}_{aug_name}_sample{sample_idx + 1}.json"
                        json_output_path = os.path.join(aug_output_dir, json_filename)
                        self.save_labelme_annotation(
                            annotation_template,
                            shapes_to_save,
                            polygons_to_save,
                            (augmented_image.shape[0], augmented_image.shape[1]),
                            os.path.basename(output_path),
                            json_output_path
                        )

                        if sample_idx == 0:
                            rel_img = os.path.relpath(
                                output_path,
                                self.output_root
                            )
                            rel_json = os.path.relpath(
                                json_output_path,
                                self.output_root
                            )
                            self.log(
                                f"Saved {aug_name} sample (total {current_samples}) to "
                                f"{rel_img} / {rel_json}"
                            )
                    except Exception as e:
                        self.log(
                            f"Error generating {aug_name} sample {sample_idx + 1}: {str(e)}"
                        )
                        continue
        except Exception as e:
            self.log(f"Error processing {image_path}: {str(e)}")

    def get_image_paths(self):
        image_extensions = [
            '*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.tiff',
            '*.JPG', '*.JPEG', '*.PNG', '*.BMP', '*.GIF', '*.TIFF'
        ]
        image_paths = []

        input_path = self.args.input_path
        if os.path.isfile(input_path):
            ext = os.path.splitext(input_path)[1].lower()
            if any(ext == target_ext[1:].lower() for target_ext in image_extensions):
                image_paths.append(input_path)
            else:
                self.log(f"Error: {input_path} is not a supported image file")

        elif os.path.isdir(input_path):
            for ext in image_extensions:
                if self.args.recursive:
                    pattern = os.path.join(input_path, '**', ext)
                    found = glob.glob(pattern, recursive=True)
                else:
                    pattern = os.path.join(input_path, ext)
                    found = glob.glob(pattern)
                found = [self.clean_path(p) for p in found if os.path.exists(p)]
                image_paths.extend(found)

        else:
            self.log(f"Error: {input_path} does not exist")

        image_paths = list(sorted(set(image_paths)))
        self.log(f"Found {len(image_paths)} valid images to process")
        return image_paths

    def run(self):
        image_paths = self.get_image_paths()

        if not image_paths:
            self.log("No images to process. Exiting.")
            return

        for img_idx, image_path in enumerate(image_paths, 1):
            self.log(f"\nProcessing image {img_idx}/{len(image_paths)}")
            self.process_image(image_path)

        self.log("\nAugmentation process completed (check logs for details)")
        self.log(f"All results saved to: {self.output_root}")

    def parse_arguments(self):
        parser = argparse.ArgumentParser(
            description='Battery Defect Image Augmentation Tool (Preserve Directory Structure)'
        )

        parser.add_argument(
            '--input_path',
            default=r'Z:\5-标注数据\CYS.250804-阳极涂布机尾外观瑕疵CCD检测ATL\检测模型\原始数据集\漏金属',
            help='Path to input image file or directory (e.g., ./input_data)'
        )
        parser.add_argument(
            '--output_dir',
            default=r'Z:\5-标注数据\CYS.250804-阳极涂布机尾外观瑕疵CCD检测ATL\检测模型\原始数据集\漏金属_数据增强',
            help='Root directory for output augmented images (e.g., ./aug_output)'
        )
        parser.add_argument(
            '--samples',
            type=int,
            default=1,
            help='Default number of augmented samples to generate per image per augmentation type (default: 1)'
        )
        parser.add_argument(
            '--augmentations',
            nargs='+',
            choices=[
                'rotate', 'brightness_contrast', 'horizontal_flip',
                'vertical_flip', 'gaussian_blur', 'random_crop',
                'gaussian_noise', 'salt_pepper_noise', 'random_scale',
                'elastic_transform'
            ],
            default=[
                'rotate', 'brightness_contrast',
                'vertical_flip', 'horizontal_flip'
            ],
            help='List of augmentations to apply (default: rotate brightness_contrast vertical_flip horizontal_flip)'
        )
        parser.add_argument(
            '--recursive',
            action='store_true',
            default=True,
            help='Process images in subdirectories recursively (default: Enabled)'
        )

        return parser.parse_args()


if __name__ == "__main__":
    try:
        augmenter = BatteryDefectAugmenter()
        augmenter.run()
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        exit(1)
