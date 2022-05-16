import csv
import argparse

from glob import glob
from pathlib import Path
from shapely.geometry import shape
from shapely import geometry
from shapely.geometry.polygon import Polygon
import json

from pathaia.util.types import Patch, Slide
from pathaia.patches.functional_api import slide_rois_no_image
from pathaia.patches import filter_thumbnail

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
    default="/home/mehdi/code/luminal/data/ba.csv",
    help="csv with the slide to use ",
)

parser.add_argument(
    "--out_file_path",
    type=Path,
    default="/data/DeepLearning/mehdi/csv/luminal_data_split.csv",
    help="name of the csv that store the split ",
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
            files = glob(str(args.raw_slide_path) + "/" + row["id"] + "-1-??-1_*")

            if files:
                if i in [x for x in range(6)] + [x for x in range(39, 43)]:
                    row["split"] = "valid"
                else:
                    row["split"] = "train"

                row["id"] = files[0]
                input_files.append(row)

                i += 1

    with open(args.out_file_path, "w") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=input_files[0].keys())
        writer.writeheader()
        for row in input_files:
            writer.writerow(row)

    for in_file_path in input_files:

        # print(in_file_path.get("ab"))

        label  = MAPPING.get(in_file_path.get("ab"))
        in_file_path = in_file_path.get("id")

        csv_file = Path(in_file_path.split(sep="/")[-1][:-4])

        out_file_path = args.outfolder / "patch_csvs" /str(args.level)/ csv_file.with_suffix(".csv")

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


        gjson = Path("/home/mehdi/code/luminal/data/geojson_lum") / csv_file.with_suffix(".geojson")
        with open(gjson, "r") as f:
            shape_dict = json.load(f)

        print(len(shape_dict))
        if not isinstance(shape_dict,list) :
            roi_shapes = [shape(shape_dict["geometry"])]
        else:
            roi_shapes = [shape(shape_r["geometry"]) for  shape_r in  shape_dict ]
            print("in")

         
        print(csv_file,label)

        with open(out_file_path, "w") as out_file:
            writer = csv.DictWriter(out_file, fieldnames=Patch.get_fields() + ["n_pos"]+["label"])
            writer.writeheader()
            for patch in patches:
                for roi_shape in roi_shapes:
                    pt1 = patch.position
                    dx = pt1.x +args.patch_size
                    dy = pt1.y + args.patch_size
                    pt2 = geometry.Point(dx,pt1.y)
                    pt3 = geometry.Point(pt1.x,dy)
                    pt4 = geometry.Point(dx,dy)
                    patch_shape = Polygon([pt1,pt2,pt4,pt3])
            
                    if roi_shape.intersects(patch_shape):
                        intersect = roi_shape.intersection(patch_shape)
                        if intersect.area/patch_shape.area>0.3:
                            row = patch.to_csv_row()
                            row["label"] =label#MAPPING.get(in_file_path.get("label"))
                            writer.writerow(row)
                            
                    else:
                        # print("junk")
                        row = patch.to_csv_row()
                        row["label"] = 2
                        writer.writerow(row)
                        
                    
    
    
 
 