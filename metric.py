import numpy as np
from scipy import interpolate
import torch
from tqdm import tqdm

from utils import restore_image


class SICAUC:
    def __init__(self, args, predict_model):
        self.args = args
        self.predict_model = predict_model

    def soft_max_(self, data, lab_ind):
        f = np.exp(data[:, lab_ind])
        a = np.sum(np.exp(data[:, :]), axis=1)
        return f/a

    def predict(self, data, lab_ind):
        # data = torch.tensor(data.clone().detach(), dtype=torch.float32).cuda()
        # image_batch = torch.tensor(image_batch, dtype=torch.float32).unsqueeze(0)
        score = self.predict_model(data).cpu().detach().numpy()
        if self.args.data in ['ADTI', 'MNIST', 'TREC']:
            score = self.soft_max_(score, lab_ind)
        return score

    def metric(self, test_dataset, shapley, reference_):
        add_list = []
        del_list = []
        for len_ind in tqdm(range(shapley.shape[0])):
            lab_ind = int(test_dataset.labels[len_ind])
            x_test = test_dataset.data[len_ind]
            if self.args.data == 'ETT':
                x_test = x_test.unsqueeze(0)
                shapley_value = shapley[len_ind, :].reshape(1, -1)
            elif self.args.data == 'MNIST':
                x_test = torch.tensor(restore_image(x_test.cpu().numpy()).reshape(1, 28, 28), dtype=torch.float32).to(self.args.device)
                shapley_value = np.kron(shapley[len_ind, :].reshape(14, 14), np.ones((2, 2)))
                shapley_value = shapley_value.reshape(1, 28, 28)
            else:
                x_test = x_test.unsqueeze(0)
                shapley_value = shapley[len_ind, :].reshape(1, -1)
            del_ = True
            gig_saliency_map = shapley_value

            reference_pred = self.predict(reference_, lab_ind)
            original_pred = self.predict(x_test, lab_ind)
            # print(reference_pred, original_pred)
            saliency_thresholds = [0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.13,
                                       0.21, 0.34, 0.5, 0.75, 1]
            s = []
            for threshold in saliency_thresholds:
                quantile = np.quantile(abs(gig_saliency_map), 1 - threshold)
                pixel_mask = abs(gig_saliency_map) >= quantile
                input_data = torch.tensor(np.multiply(x_test.cpu().numpy(), pixel_mask), dtype=torch.float32).to(self.args.device)
                score_s = self.predict(input_data, lab_ind)
                score_s = (score_s-reference_pred)/(original_pred-reference_pred)
                score_s = np.clip(score_s, 0.0, 1.0)
                s.append(score_s)


            interp_func = interpolate.interp1d(x=np.array(saliency_thresholds), y=np.array(s).T)
            curve_x = np.linspace(start=0.0, stop=1.0, num=1000,
                                    endpoint=False)
            curve_y = np.asarray([interp_func(x) for x in curve_x])
            curve_x = np.append(curve_x, 1.0)
            curve_y = np.append(curve_y, 1.0)
            auc_add = np.trapz(curve_y, curve_x)
            # print('add', auc_add)

            # saliency_thresholds = [0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.13,
            #                            0.21, 0.34, 0.5, 0.75, 1]
            s = []
            for threshold in saliency_thresholds:
                quantile = np.quantile(abs(gig_saliency_map), 1 - threshold)
                pixel_mask = abs(gig_saliency_map) <= quantile
                input_data = torch.tensor(np.multiply(x_test.cpu().numpy(), pixel_mask), dtype=torch.float32).to(self.args.device)
                score_s = self.predict(input_data, lab_ind)
                score_s = (score_s-reference_pred)/(original_pred-reference_pred)
                score_s = np.clip(score_s, 0.0, 1.0)
                s.append(score_s)

            interp_func = interpolate.interp1d(x=np.array(saliency_thresholds), y=np.array(s).T)
            curve_x = np.linspace(start=0.0, stop=1.0, num=1000,
                                    endpoint=False)
            curve_y = np.asarray([interp_func(x) for x in curve_x])
            curve_x = np.append(curve_x, 1.0)
            curve_y = np.append(curve_y, 0.0)
            auc_del = np.trapz(curve_y, curve_x)
            # print('del', auc_del)
            if auc_add>=auc_del:
                add_list.append(auc_add)
                del_list.append(auc_del)
            else:
                add_list.append(auc_del)
                del_list.append(auc_add)

        print(np.array(add_list).mean())
        print(np.array(del_list).mean())