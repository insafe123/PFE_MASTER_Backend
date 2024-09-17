import vtk
import os
from meshsegnet import *
import vedo
from losses_and_metrics_for_mesh import *
from scipy.spatial import distance_matrix
import shutil
import time


def colorate_d(file_path):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(file_path)
    reader.Update()
    polyData = reader.GetOutput()

    # Define the colors for each class
    color_map = {
        0: (0.5, 0.5, 0.5), # Gray
        1: (0, 1, 0.5), # Teal
        2: (0, 0, 1),    # Blue
        3: (0.5, 0.5, 0), # Olive
        4: (1, 0, 1),    # Magenta
        5: (0, 0.5, 0.5), # Turquoise
        6: (1, 0.5, 0),  # Orange
        7: (0.5, 1, 0),  # Lime
        8: (1, 0, 0),  # Red
        9: (0, 1, 0),    # Green
        10: (1, 0, 0.5),  # Pink
        11: (1, 1, 0),  # Yellow
        12: (0.5, 0, 0.5), # Plum
        13: (0, 1, 1),  # Cyan
        14: (0.5, 0, 1),  # Purple
    }


    # Create a lookup table to map the class IDs to colors
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(len(color_map))
    lut.Build()

    for class_id, color in color_map.items():
        lut.SetTableValue(class_id, color[0], color[1], color[2],  1.0)

    # Set the lookup table on the poly data
    polyData.GetCellData().SetScalars(polyData.GetCellData().GetArray("Label"))
    polyData.GetCellData().GetScalars().SetLookupTable(lut)


    # filename = "downsampled_colored_file.vtp"
    # filepath = os.path.join(pathsave, filename)
    #
    # # Create the output directory if it doesn't exist
    # if not os.path.exists(pathsave):
    #     os.makedirs(pathsave)
    #
    # writer = vtk.vtkXMLPolyDataWriter()
    # writer.SetFileName(filepath)
    # writer.SetInputData(polyData)
    # writer.Write()

    # Write the file using vtkXMLPolyDataWriter
    # output_filename = os.path.join(pathsave, 'downsampled_colored_file.vtp')
    # writer = vtk.vtkXMLPolyDataWriter()
    # writer.SetFileName(output_filename)
    # writer.SetInputData(polyData)
    # writer.Write()

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName("downsampled_colored_file.vtp")
    writer.SetInputData(polyData)
    writer.Write()

def colorate_d_r(file_path):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(file_path)
    reader.Update()
    polyData = reader.GetOutput()

    # Define the colors for each class
    color_map = {
        0: (0.5, 0.5, 0.5), # Gray
        1: (0, 1, 0.5), # Teal
        2: (0, 0, 1),    # Blue
        3: (0.5, 0.5, 0), # Olive
        4: (1, 0, 1),    # Magenta
        5: (0, 0.5, 0.5), # Turquoise
        6: (1, 0.5, 0),  # Orange
        7: (0.5, 1, 0),  # Lime
        8: (1, 0, 0),  # Red
        9: (0, 1, 0),    # Green
        10: (1, 0, 0.5),  # Pink
        11: (1, 1, 0),  # Yellow
        12: (0.5, 0, 0.5), # Plum
        13: (0, 1, 1),  # Cyan
        14: (0.5, 0, 1),  # Purple
    }


    # Create a lookup table to map the class IDs to colors
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(len(color_map))
    lut.Build()

    for class_id, color in color_map.items():
        lut.SetTableValue(class_id, color[0], color[1], color[2],  1.0)

    # Set the lookup table on the poly data
    polyData.GetCellData().SetScalars(polyData.GetCellData().GetArray("Label"))
    polyData.GetCellData().GetScalars().SetLookupTable(lut)


    # filename = "downsampled_colored_file.vtp"
    # filepath = os.path.join(pathsave, filename)
    #
    # # Create the output directory if it doesn't exist
    # if not os.path.exists(pathsave):
    #     os.makedirs(pathsave)
    #
    # writer = vtk.vtkXMLPolyDataWriter()
    # writer.SetFileName(filepath)
    # writer.SetInputData(polyData)
    # writer.Write()

    # Write the file using vtkXMLPolyDataWriter
    # output_filename = os.path.join(pathsave, 'downsampled_colored_file.vtp')
    # writer = vtk.vtkXMLPolyDataWriter()
    # writer.SetFileName(output_filename)
    # writer.SetInputData(polyData)
    # writer.Write()

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName("downsampled_refined_colored_file.vtp")
    writer.SetInputData(polyData)
    writer.Write()

def colorate_r(file_path):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(file_path)
    reader.Update()
    polyData = reader.GetOutput()

    # Define the colors for each class
    color_map = {
        0: (0.5, 0.5, 0.5), # Gray
        1: (0, 1, 0.5), # Teal
        2: (0, 0, 1),    # Blue
        3: (0.5, 0.5, 0), # Olive
        4: (1, 0, 1),    # Magenta
        5: (0, 0.5, 0.5), # Turquoise
        6: (1, 0.5, 0),  # Orange
        7: (0.5, 1, 0),  # Lime
        8: (1, 0, 0),  # Red
        9: (0, 1, 0),    # Green
        10: (1, 0, 0.5),  # Pink
        11: (1, 1, 0),  # Yellow
        12: (0.5, 0, 0.5), # Plum
        13: (0, 1, 1),  # Cyan
        14: (0.5, 0, 1),  # Purple
    }


    # Create a lookup table to map the class IDs to colors
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(len(color_map))
    lut.Build()

    for class_id, color in color_map.items():
        lut.SetTableValue(class_id, color[0], color[1], color[2],  1.0)

    # Set the lookup table on the poly data
    polyData.GetCellData().SetScalars(polyData.GetCellData().GetArray("Label"))
    polyData.GetCellData().GetScalars().SetLookupTable(lut)

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName("refined_colored_file.vtp")
    writer.SetInputData(polyData)
    writer.Write()

