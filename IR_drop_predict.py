import os
import argparse
import numpy as np
import torch
from IR_drop_model import IRdropModel
import cv2
import matplotlib.pyplot as plt
import pandas as pd

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class IRDropPrediction():
    def __init__(self,datapath,features,model_path,device):
        super(IRDropPrediction, self).__init__()
        self.datapath = datapath
        self.FeaturePathList = features
        self.feature = self.data_process(self.FeaturePathList).unsqueeze(0).to(device)
        self.model = IRdropModel(in_channel=24,device=device).to(device)
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        self.device = device

    def resize_cv2(self,input):
        output = cv2.resize(input, (256, 256), interpolation=cv2.INTER_AREA)
        return output

    def std(self,input):
        if input.max() == 0:
            return input
        else:
            result = (input - input.min()) / (input.max() - input.min())
            return result

    def data_process(self,FeaturePathList):
        features = []
        for feature_name in FeaturePathList:
            name = os.listdir(os.path.join(self.datapath,feature_name))[0]
            feature = np.load(os.path.join(self.datapath,feature_name,name))
            if feature_name == "power_t":
                for i in range(20):
                    slice = feature[i, :, :]
                    features.append(torch.as_tensor(self.std(self.resize_cv2(slice))))
            else:
                feature = self.std(self.resize_cv2(feature.squeeze()))
                features.append(torch.as_tensor(feature))
        features = torch.stack(features).type(torch.float32)
        return features

    def find_irdrop_coord_and_value(self,tensor, threshold):
        indices = torch.where(tensor >= threshold)
        values = self.std(tensor[indices])
        return np.array(list((indices[1].tolist(), indices[0].tolist(),values.tolist()))).T

    def Prediction(self, irdrop_threshold):
        self.irdrop_threshold = irdrop_threshold
        if self.device != 'cpu':
            with torch.cuda.amp.autocast():
                self.pred = self.model(self.feature)
                self.pred = self.model.sigmoid(self.pred)
        if self.device == 'cpu':
            self.pred = self.model(self.feature)
            self.pred = self.model.sigmoid(self.pred)
        self.pred_coord = self.find_irdrop_coord_and_value(self.pred[0,0], threshold=irdrop_threshold)
        self.pred_coord = pd.DataFrame(self.pred_coord,columns=['x','y','congestion'])
        return self.pred, self.pred_coord

    def ShowFig(self,fig_save_path):
        if fig_save_path is None:
            raise ValueError("Figure save path is not specified clear.")
        plt.imshow(np.zeros(shape=self.pred[0,0].shape))
        # plt.imshow(self.pred[0, 0].detach().cpu().numpy())
        plt.title(f"IR Drop Prediction > {self.irdrop_threshold}")
        pts = plt.scatter(x=self.pred_coord['x'],y=self.pred_coord['y'],c=self.pred_coord['congestion'],cmap='jet',s=5)
        # plt.legend([pts],["IR Drop"])
        plt.colorbar()
        plt.savefig(f"{fig_save_path}/IRDrop_{self.irdrop_threshold}.png")
        plt.show()

    def save(self,output_path):
        np.save(f"{output_path}/PredArray",self.pred[0,0].detach().cpu().numpy())
        self.pred_coord.to_csv(f"{output_path}/PredCoord.csv")


def parse_args():
    description = "Input the Path for Prediction"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data_path", default="./data", type=str, help='The path of the data file')
    parser.add_argument("--fig_save_path", default="./save_img", type=str, help='The path you want to save fingue')
    parser.add_argument("--weight_path", default="./model_weight/irdrop_weights.pt", type=str, help='The path of the model weight')
    parser.add_argument("--output_path", default="./output", type=str, help='The path of the model weight')
    parser.add_argument("--irdrop_threshold", default=0.1, type=float, help='irdrop_threshold [0,1]')
    parser.add_argument("--device", default='cpu', type=str, help='If you have gpu type "cuda" will be faster!!')
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    import time
    start = time.time()
    args = parse_args()
    feature_list = ['power_i', 'power_s','power_sca', 'power_all', 'power_t']

    predictionSystem = IRDropPrediction(datapath=args.data_path,features=feature_list,
                                model_path=args.weight_path,device=args.device)
    pred,pred_coord = predictionSystem.Prediction(irdrop_threshold=args.irdrop_threshold)
    predictionSystem.save(args.output_path)
    end = time.time()
    print("cost time：%f sec" % (end - start))
    if args.fig_save_path !=None:
        predictionSystem.ShowFig(args.fig_save_path)

