import argparse
import csv
import json

from glob import glob
from pathaia.util.types import Patch, Slide
from pathaia.patches.functional_api import slide_rois_no_image
from pathaia.patches import filter_thumbnail
from pathlib import Path
from shapely.geometry import shape
from shapely import geometry
from shapely.geometry.polygon import Polygon


MAPPING = {
    "luminal A": 0,
    "luminal B": 1,
}

# Init the parser
parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)


# Add path to the config file to the command line arguments
parser.add_argument(
    "--outfolder",
    type=Path,
    default="/data/DeepLearning/mehdi/csv_annot",
    help="folder storing csvs",
)

parser.add_argument(
    "--patch_size",
    type=int,
    default=1024,
    help="size of the patches",
)
parser.add_argument(
    "--level",
    type=int,
    default=0,
    help=" ",
)
parser.add_argument(
    "--overlap",
    type=int,
    default=0,
    help=" ",
)
parser.add_argument(
    "--overwrite",
    action="store_true",
    help=" .",
)

parser.add_argument(
    "--raw_slide_path",
    type=Path,
    default="/media/AprioricsSlides/",
    help="where to get the slides",
)
parser.add_argument(
    "--path_to_file",
    type=Path,
    default="/data/DeepLearning/mehdi/csv/ab_comp.csv",
    help="csv with the slide to use ",
)

parser.add_argument(
    "--out_file_path",
    type=Path,
    default="/data/DeepLearning/mehdi/csv/luminal_data_split.csv",
    help="name of the csv that store the split ",
)

parser.add_argument(
    "--area_intercect_percentage",
    type=int,
    default=0.9,
    help="how much overlay you want to allow between the anotation and the patch ",
)
parser.add_argument(
    "--annotation",
    type=bool,
    default=False,
    help="use an annotation",
)


args = parser.parse_args()


if __name__ == "__main__":
    input_files = []
    i = 0

    interval = -int(args.overlap * args.patch_size)

    with open(args.path_to_file, "r") as patch_file:
        # reader = csv.DictReader(patch_file)
        reader = csv.DictReader(patch_file, delimiter=";")
        for row in reader:
            new_row = {}
            files = glob(str(args.raw_slide_path) + "/" + row["pseudo"] + "-1-??-1_*")

            if not files:

                files = glob(
                    str(args.raw_slide_path) + "/" + row["pseudo"] + "-1-?-1_*"
                )

            if files:
                print(files)

                new_row["id"] = files[0]
                new_row["ab"] = row["ab"]
                input_files.append(new_row)

                i += 1

    with open(args.out_file_path, "w") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=input_files[0].keys())
        writer.writeheader()
        for row in input_files:
            writer.writerow(row)

    for in_file_path in input_files:

        # print(in_file_path.get("ab"))

        label = MAPPING.get(in_file_path.get("ab"))
        in_file_path = in_file_path.get("id")

        csv_file = Path(in_file_path.split(sep="/")[-1][:-4])

        out_file_path = (
            args.outfolder
            / "patch_csvs"
            / str(args.level)
            / str(args.patch_size)
            / csv_file.with_suffix(".csv")
        )

        # out_file_path = outfolder / in_file_path.relative_to(
        #     args.slidefolder
        # ).with_suffix(".csv")
        if not args.overwrite and out_file_path.exists():
            continue
        if not out_file_path.parent.exists():
            out_file_path.parent.mkdir(parents=True)

        slide = Slide(in_file_path, backend="cucim")

        # print(in_file_path.stem)

        patches = slide_rois_no_image(
            slide,
            args.level,
            psize=args.patch_size,
            interval=interval,
            slide_filters=[filter_thumbnail],
            thumb_size=2000,
        )

        if args.annotations:

            gjson = Path(
                "/media/AprioricsSlides/annot tum lum A vs B"
            ) / csv_file.with_suffix(".geojson")
            with open(gjson, "r") as f:
                shape_dict = json.load(f)

            if "features" in shape_dict:
                shape_dict = shape_dict.get("features")
            print(len(shape_dict))
            if not isinstance(shape_dict, list):
                shape_ = shape_dict.get("geometry")

                roi_shapes = [shape(shape_)]
            else:
                roi_shapes = [shape(shape_r["geometry"]) for shape_r in shape_dict]
                print("in")

        print(csv_file, label)

        with open(out_file_path, "w") as out_file:
            writer = csv.DictWriter(
                out_file, fieldnames=Patch.get_fields() + ["n_pos"] + ["label"]
            )
            writer.writeheader()
            for patch in patches:
                if args.annotations:

                    for num_shape, roi_shape in enumerate(roi_shapes):
                        pt1 = patch.position
                        dx = pt1.x + args.patch_size
                        dy = pt1.y + args.patch_size
                        pt2 = geometry.Point(dx, pt1.y)
                        pt3 = geometry.Point(pt1.x, dy)
                        pt4 = geometry.Point(dx, dy)
                        patch_shape = Polygon([pt1, pt2, pt4, pt3])

                        if roi_shape.intersects(patch_shape):
                            intersect = roi_shape.intersection(patch_shape)
                            if (
                                intersect.area / patch_shape.area
                                > args.area_intercect_percentage
                            ):
                                row = patch.to_csv_row()
                                row[
                                    "label"
                                ] = label  # MAPPING.get(in_file_path.get("label"))
                                writer.writerow(row)
                                break

                            else:
                                # print("junk")
                                row = patch.to_csv_row()
                                row["label"] = 2
                                writer.writerow(row)
                                break
                        else:
                            if num_shape == len(roi_shapes) - 1:
                                # print("junk")
                                row = patch.to_csv_row()
                                row["label"] = 2
                                writer.writerow(row)
                    else:
                        row = patch.to_csv_row()
                        row["label"] = label  # MAPPING.get(in_file_path.get("label"))
                        writer.writerow(row)
