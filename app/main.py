#!/usr/bin/python3
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from typing import List
import pyvips
from PIL import Image, ImageOps
from .photo_lib import PhotoDict

app = FastAPI()
app.mount("/static", StaticFiles(directory="disney_b"), name="static")
app.mount("/static_sm", StaticFiles(directory="resized_album"), name="static_sm")
app.mount("/static_faces", StaticFiles(directory="faces_gallery"), name="static_faces")
templates = Jinja2Templates(directory="templates/")

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/gallery", response_class=HTMLResponse)
async def gallery_get(request: Request):
    # selected_user_id = "Type a face id"
    selected_user_id, user_rec_dict = get_userid_files(-1)
    return get_gallery(request, selected_user_id, user_rec_dict)


@app.post("/gallery", response_class=HTMLResponse)
async def gallery_post(request: Request, user_id: int = Form(...)):
    selected_user_id, user_rec_dict = get_userid_files(user_id)
    return get_gallery(request, selected_user_id, user_rec_dict)


def get_gallery(request, selected_user_id, user_rec_dict):
    return templates.TemplateResponse(
        "gallery.html",
        context={
            "request": request,
            "selected_user_id": selected_user_id,
            "user_rec_dict": user_rec_dict,
            "userids": get_faces_list(),
        },
    )


# Album specific logic below; serving logic above
photo_dict = None
user_rec_dict = {}
# user_rec_dict: dict[int, List[str]] = {} 


def get_photo_dict():
    global photo_dict
    if photo_dict is None:
        input_filelist = sorted(Path.cwd().rglob("disney_b/*.*"))
        photo_dict = PhotoDict(input_filelist)
        # Let's resize the album when we serve this for the first time
        resize_album(input_filelist, "resized_album")
        create_faces_album(photo_dict, "faces_gallery")
        # Let's precompute the recommendations when we serve the first time.
        for _, x in enumerate(get_faces_list()):
            get_userid_files(x)
    return photo_dict


def get_userid_files(user_id: int):
    global user_rec_dict
    photo_dict = get_photo_dict()
    if user_id in user_rec_dict:
        filelist = user_rec_dict[user_id]
    else:
        filelist = [file.name for file in photo_dict.sort_by_userid(user_id)]
        user_rec_dict[user_id] = filelist
    return user_id, user_rec_dict


def get_faces_list():
    # TODO: Remove unclassified
    photo_dict = get_photo_dict()
    return sorted(set(photo_dict.face_labels))


# Resizing functions
def create_faces_album(photo_dict: PhotoDict, dest_dir):
    for _key, photofile in photo_dict.dict.items():
        for idx, crop_location in enumerate(photofile.face_locations):
            # output_filename = Path.cwd() / dest_dir / photofile.filename.name
            filename = (
                "face_"
                + str(photo_dict.face_labels[photofile.face_indices[idx]])
                + ".jpg"
            )
            output_filename = Path.cwd() / dest_dir / filename
            create_face_thumbnail(
                str(photofile.filename), str(output_filename), crop_location
            )


def create_face_thumbnail(input_filename, output_filename, crop_location):
    # image_vips = pyvips.Image.new_from_file(input_filename)
    image_vips = Image.open(input_filename)
    image_vips_rot = ImageOps.exif_transpose(image_vips)
    x0, y0, x1, y1 = crop_location

    x0 = 0 if x0 < 0 else x0
    y0 = 0 if y0 < 0 else y0
    # print(
    #     output_filename,
    #     x0,
    #     y0,
    #     x1,
    #     y1,
    #     image_vips.width,
    #     image_vips.height,
    # )

    width = x1 - x0
    height = y1 - y0
    if width > 0 and height > 0:
        cropped = image_vips_rot.crop((y0, x0, y1, x1))
    else:
        print("not cropping")
        cropped = image_vips_rot

    cropped.save(output_filename)


def resize_album(files, dest_dir):
    for file in files:
        output_filename = Path.cwd() / dest_dir / file.name
        create_thumbnail(str(file), str(output_filename))


def create_thumbnail(input_filename, output_filename):
    image = pyvips.Image.new_from_file(input_filename, shrink=4)
    image.write_to_file(output_filename)


# HACK - Remove later; exists only to pre-compute on start (& not on first http request)
get_userid_files(-1)