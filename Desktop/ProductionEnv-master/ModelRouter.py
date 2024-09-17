import json
import os
from io import BytesIO
from typing import Dict

from fastapi import APIRouter, UploadFile, File, Path, Response
import ipyvolume as ipv
from fastapi.responses import StreamingResponse, FileResponse
import pyvista as pv
from starlette.responses import JSONResponse, PlainTextResponse

import Colorate
import segmenter
from pathlib import Path
from fastapi.responses import JSONResponse

model_route = APIRouter()


def process_upload(file: UploadFile) -> str:
    file_path = Path(os.getcwd()) / file.filename
    with file_path.open("wb") as buffer:
        buffer.write(file.file.read())
    return str(file_path.resolve())


@model_route.post("/upload-file-predict-labels/")
async def upload_file(file: UploadFile = File(...)):
    file_path = process_upload(file)
    segmenter.predict_labels(file_path)
    path = os.path.join(os.getcwd(), 'output', 'out_refined.json')
    with open(path, mode="r") as f:
        contents = f.read()
        data = json.loads(contents)
    #
    # # Format the predicted labels as index-label pairs
    # data_list = []
    # for key, value in data_dict.items():
    #     data_list.append('{} {}'.format(key, value))
    #
    # # Return the formatted index-label pairs as a string
    # response_str = '\n'.join(data_list)
    # return PlainTextResponse(response_str)

    #
    # # Format the predicted labels as index-label pairs in an array
    # data_list = []
    # for key, value in data_dict.items():
    #     data_list.append([int(key), int(value)])
    #
    # # Return the formatted index-label pairs as an array
    # response_dict = {"result": data_list}
    # return JSONResponse(content=response_dict)

        # data = json.loads(contents)
        # labels = JSONResponse(content=data)

    # return {data_dict}
    return JSONResponse(content=data)

    #     data = json.loads(contents)
    # return JSONResponse(content=data)

    # file_like = open(path, mode="rb")
    # return StreamingResponse(
    #     file_like,
    #     media_type="application/octet-stream",
    #     headers={
    #         "Content-Disposition": f"attachment; filename=out_downsampling_refined.json"
    #     },
    # )


@model_route.post("/upload-file-segment-refined/")
async def upload_file(file: UploadFile = File(...)):
    file_path = process_upload(file)
    segmenter.segment(file_path)
    path = os.path.join(os.getcwd(), 'output', 'out_refined.vtp')
    Colorate.colorate_r(path)
    path1 = os.path.join(os.getcwd(), "refined_colored_file.vtp")
    # reader = vtk.vtkXMLPolyDataReader()
    # reader.SetFileName(path1)
    # reader.Update()
    # ply_writer = vtk.vtkPLYWriter()
    # ply_writer.SetFileName("refined_colored_file.ply")
    # ply_writer.SetInputData(reader.GetOutput())
    # ply_writer.Write()

    file_like = open(path1, mode="rb")
    return StreamingResponse(
        file_like,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f"attachment; filename=refined_colored_file.vtp"
        },
    )



@model_route.post("/upload-file-segment-downsampling/")
async def upload_file(file: UploadFile = File(...)):
    file_path = process_upload(file)
    segmenter.segment_d(file_path)
    path = os.path.join(os.getcwd(), 'output', 'out_downsampling.vtp')
    Colorate.colorate_d(path)
    path1 = os.path.join(os.getcwd(), 'downsampled_colored_file.vtp')
    file_like = open(path1, mode="rb")

    return StreamingResponse(
        file_like,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f"attachment; filename=downsampled_colored_file.vtp"
        },
    )


@model_route.post("/upload-file-segment-downsampling-refined/")
async def upload_file(file: UploadFile = File(...)):
    file_path = process_upload(file)
    segmenter.segment_d_r(file_path)
    path = os.path.join(os.getcwd(), 'output', 'out_downsampling_refined.vtp')
    #Colorate.colorate_d_r(path)
    #path1 = os.path.join(os.getcwd(), 'downsampled_refined_colored_file.vtp')
    file_like = open(path, mode="rb")
    #
    # mesh = pv.read(path1)
    # # Create a PyVista plotter and add the mesh to it
    # plotter = pv.Plotter()
    # plotter.add_mesh(mesh)
    #
    # # Show the plotter on screen
    # plotter.show()
    #
    # # Take a screenshot of the plotter at a specified angle
    # plotter.camera_position = [(1, 1, 1), (0, 0, 0), (0, 0, 1)]
    # screenshot = plotter.screenshot()
    #
    # # Return the screenshot as a response
    # return Response(content=screenshot, media_type="image/png")
    return StreamingResponse(
        file_like,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f"attachment; filename=out_downsampled_refined_file.vtp"
        },
    )

@model_route.get('/view_object')
async def view_object():
    mesh = pv.read("refined_colored_file.vtp")

    plotter = pv.Plotter()
    plotter.add_mesh(mesh)
    plotter.background_color = 'white'
    plotter.show()

@model_route.post("/upload-file-pyvista/")
async def upload_file(file: UploadFile = File(...)):
    file_path = process_upload(file)
    # path1 = os.path.join(os.getcwd(), 'downsampled_colored_file.vtp')
    #file_like = open(path1, mode="rb")
    mesh = pv.read(file_path)
    # # Convert the PyVista mesh to an ipyvolume mesh
    # ipv_mesh = ipv.PyVistaMesh(mesh)
    #
    # # Create an ipyvolume figure and add the mesh to it
    # fig = ipv.figure()
    # ipv_mesh = ipv.quickvolshow(mesh, lighting=True, level=[0.2, 0.6, 0.9])
    #
    # # Convert the figure to a PNG image and return it as a response
    # png = BytesIO()
    # ipv.savefig(png, format='png')
    # png.seek(0)
    # return Response(content=png.read(), media_type="image/png")
    # Create a PyVista plotter and add the mesh to it
    plotter = pv.Plotter()
    plotter.add_mesh(mesh)

    # Show the plotter on screen
    plotter.show()

    # Take a screenshot of the plotter at a specified angle
    plotter.camera_position = [(1, 1, 1), (0, 0, 0), (0, 0, 1)]
    screenshot = plotter.screenshot()

    # Return the screenshot as a response
    return Response(content=screenshot, media_type="image/png")
    # return StreamingResponse(
    #     file_like,
    #     media_type="application/octet-stream",
    #     headers={
    #         "Content-Disposition": f"attachment; filename=downsampled_colored_file.vtp"
    #     },
    # )


@model_route.get("/vtp-file")
def get_vtp_file():
    path = os.path.join(os.getcwd(), 'output', 'out_downsampling_refined.vtp')
    Colorate.colorate_d_r(path)
    mesh = pv.read(path)

    plotter = pv.Plotter()
    plotter.add_mesh(mesh, scalars='Label')
    plotter.background_color = 'white'
    plotter.show()
    return FileResponse(path, media_type="application/octet-stream", filename="downsampled_refined_colored_file.vtp")
