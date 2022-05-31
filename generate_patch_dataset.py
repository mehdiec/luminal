import argparse
import csv

from glob import glob
from pathaia.util.types import Patch, Slide
from pathaia.patches import filter_thumbnail
from pathaia.patches.functional_api import slide_rois_no_image
from pathlib import Path


# Init the parser
parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)


# Add path to the config file to the command line arguments
parser.add_argument(
    "--outfolder",
    type=Path,
    default="/data/DeepLearning/mehdi/csv",
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

        in_file_path = in_file_path.get("id")

        csv_file = Path(in_file_path.split(sep="/")[-1][:-4])

        out_file_path = (
            args.outfolder
            / "patch_csvs"
            / str(args.level)
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

        with open(out_file_path, "w") as out_file:
            writer = csv.DictWriter(out_file, fieldnames=Patch.get_fields() + ["n_pos"])
            writer.writeheader()
            for patch in patches:

                row = patch.to_csv_row()
                writer.writerow(row)
