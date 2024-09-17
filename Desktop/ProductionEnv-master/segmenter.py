import Colorate
import DownsamplingRefined
import PredictLabels
import Refined
import Downsampling


def segment(path):
    #checkArgs(path)
    model = 'C:/Users/Admin/PycharmProjects/ProductionEnv/models/Mesh_Segementation_MeshSegNet_15_classes_10samples_best.tar'
    #predict.predict(model, path)
    Refined.predict(model, path)

# def checkArgs( path_file):
#     if path_file == None:
#         raise ValueError("--path-file must be filled with path to file")
#
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Process teeth segmentation.')
#     parser.add_argument('--path-file', type=str, help='path to VTP file')
#     args = parser.parse_args()
#     path_file = args.path_file
#     segment(path_file)


def segment_d(path):
    model = 'C:/Users/Admin/PycharmProjects/ProductionEnv/models/Mesh_Segementation_MeshSegNet_15_classes_10samples_best.tar'
    #model = 'C:/Users/Admin/PycharmProjects/ProductionEnv/models/MeshSegNet_Max_15_classes_72samples_lr1e-2_best (1).tar'
    Downsampling.predict(model, path)


def segment_d_r(path):
    model = 'C:/Users/Admin/PycharmProjects/ProductionEnv/models/Mesh_Segementation_MeshSegNet_15_classes_10samples_best.tar'
    DownsamplingRefined.predict(model, path)

def predict_labels(path):
    model = 'C:/Users/Admin/PycharmProjects/ProductionEnv/models/Mesh_Segementation_MeshSegNet_15_classes_10samples_best.tar'
    PredictLabels.predict(model, path)
