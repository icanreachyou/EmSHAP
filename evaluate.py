import torch
import numpy as np
from scipy.special import comb, perm
from utils import *
class Evaluation:
    def __init__(self, args, proposal_network, energy_network):
        self.args = args
        self.proposal_network = proposal_network
        self.energy_network = energy_network
        self.device = args.device
        self.context_size = args.context_units
        self.number_features = args.num_features
        self.number_seq = args.num_seq
        self.sample_num = args.sample_num
    def impute(self, proposal_network, energy_network, data, mask_):
        if len(data.shape) == 2:
            data = data.unsqueeze(0)
            mask_ = mask_[np.newaxis, :, :]
        x_o = data * (torch.Tensor(mask_).cuda())
        query = torch.Tensor(1 - mask_).cuda()
        x_u = data * query
        x_u = x_u.cuda()
        with torch.no_grad():
            if self.args.if_energy == False:
                proposal_dist_detach, context = proposal_network(data, torch.Tensor(mask_).cuda())
                proposal_mean = proposal_dist_detach.mean * query.reshape(proposal_dist_detach.mean.shape)
                return x_o + proposal_mean * (torch.Tensor(1-mask_).cuda())
            else:
                proposal_dist_detach, context = proposal_network(data, torch.Tensor(mask_).cuda())
                if self.args.use_proposal_mean:
                    proposal_samples = proposal_dist_detach.mean.unsqueeze(1)  # [B,1,seq,F]
                else:
                    proposal_samples = proposal_dist_detach.sample((self.args.sample_num,)).unsqueeze(1)  # [K,B,1,seq,F]
                    proposal_samples = proposal_samples.permute(1, 0, 2, 3, 4)  # [B,K,1,seq,F]
                    proposal_samples_proposal_ll = proposal_dist_detach.log_prob(proposal_samples)

                    x_u_and_samples = torch.cat([x_u.unsqueeze(0).unsqueeze(0), proposal_samples],
                                                dim=1)  # [B,1+K,1,seq,F]

                    B, K_plus1, bat, S, F = x_u_and_samples.shape
                    x_u_i = x_u_and_samples.reshape(-1)  # flatten
                    u_i = torch.arange(F, device=data.device).repeat(K_plus1, bat, S, 1).reshape(-1)

                    tiled_context = context.repeat(1, K_plus1, 1, 1, 1)  # [B,K+1,...]
                    tiled_context = tiled_context.reshape(-1, self.args.context_units)

                    negative_energies = energy_network(x_u_i, u_i, tiled_context)
                    negative_energies = negative_energies.view(B, K_plus1, bat, S, F) * query

                    proposal_samples_unnorm_energy_ll = negative_energies[:, 1:]  # [B,K,S,F]

                    proposal_samples_log_ratios = proposal_samples_unnorm_energy_ll - proposal_samples_proposal_ll

                    is_weights = torch.softmax(proposal_samples_log_ratios, dim=1)  # [B,K,S,F]
                    energy_mean = (is_weights * proposal_samples).sum(dim=1) * query

                return x_o + energy_mean * (torch.Tensor(1 - mask_).cuda())


    def binary(self, x, bits):
        mask = 2 ** torch.arange(bits) # .to(x.device, x.dtype)
        return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()

    def sample_strategy(self, M, sample):
        a = np.random.randint(0, M, size=(sample, 1))
        rows, cols = sample, M-1
        b = np.zeros((rows, cols), dtype=int)
        for i in range(rows // 2):
            ones_count = a[i, 0]
            ones_indices = np.random.choice(cols, size=ones_count, replace=False)
            b[i, ones_indices] = 1
            b[rows - i - 1, ones_indices] = 1

            b[rows - i - 1, :] = ~b[rows - i - 1, :] + 2

        return torch.tensor(b).byte()

    @torch.no_grad()
    def brute_force_shapley_sample(self, f, x, sample, test_dataset, lab):
        M = x.shape[-1]

        shap_index = torch.arange(M)

        mask = self.sample_strategy(M, sample)
        # mask_dec = torch.arange(0, 2 ** (M - 1))
        # mask = binary(mask_dec, M - 1)

        shapley_value = torch.zeros((1, len(shap_index)))
        for idx, feature_index in enumerate(shap_index):
            shapley_value[:, idx] = self.sub_brute_force_shapley(f, x, mask, feature_index, M, test_dataset, lab).squeeze(dim=0)

        return shapley_value

    @torch.no_grad()
    def sub_brute_force_shapley(self, f, x, mask, feature_index, M, test_dataset, lab):
        mask = torch.cat((torch.zeros((1, mask.shape[1])).byte(), mask), dim=0)
        mask = torch.cat((mask, torch.ones((1, mask.shape[1])).byte()), dim=0)
        set0 = torch.cat((mask, torch.zeros((mask.shape[0], 1)).byte()), dim=1)
        set1 = torch.cat((mask, torch.ones((mask.shape[0], 1)).byte()), dim=1)
        # set01 = torch.cat((set0, set1), dim=0)
        set0[:, [feature_index, -1]] = set0[:, [-1, feature_index]]
        set1[:, [feature_index, -1]] = set1[:, [-1, feature_index]]
        S = set0.sum(dim=1)

        weights = 1. / torch.from_numpy(comb(x.shape[-1]-1, S)).type(torch.float).to(self.device)

        set0_x = []
        set1_x = []
        for impute_indx in range(set0.shape[0]):
            impute_data_proposal0 = self.impute(self.proposal_network, self.energy_network, x.to(self.device),
                                                         torch.broadcast_to(set0[impute_indx, :].reshape(1, -1), size=x.shape).to(self.device))
            impute_data_proposal1 = self.impute(self.proposal_network, self.energy_network, x.to(self.device),
                                                         torch.broadcast_to(set1[impute_indx, :].reshape(1, -1), size=x.shape).to(self.device))
            set0_x.append(impute_data_proposal0.squeeze(0).cpu().numpy())
            set1_x.append(impute_data_proposal1.squeeze(0).cpu().numpy())

        if self.args.data == 'ADTI':
            set0_x = torch.tensor(np.array(set0_x), dtype=torch.float32).to(self.device)
            set1_x = torch.tensor(np.array(set1_x), dtype=torch.float32).to(self.device)
            f_set0 = f(set0_x.squeeze(1))
            f_set1 = f(set1_x.squeeze(1))
            inds = test_dataset.labels[lab].cpu().numpy()
            shapley_value = 1. / M * weights.unsqueeze(dim=0).mm((f_set1[:, inds] + 1e-10).reshape(-1, 1)
                                                                 - (f_set0[:, inds] + 1e-10).reshape(-1, 1))
        if self.args.data == 'ETT':
            set0_x = torch.tensor(np.array(set0_x), dtype=torch.float32).to(self.device)
            set1_x = torch.tensor(np.array(set1_x), dtype=torch.float32).to(self.device)
            f_set0 = f(set0_x)
            f_set1 = f(set1_x)
            inds = test_dataset.labels[lab].cpu().numpy()
            shapley_value = 1. / M * weights.unsqueeze(dim=0).mm((f_set1[:, inds] + 1e-10).reshape(-1, 1)
                                                                 - (f_set0[:, inds] + 1e-10).reshape(-1, 1))
        if self.args.data == 'MNIST':
            set0_x = torch.tensor(np.array([restore_image(set0_x[i]) for i in range(len(set0_x))]),
                                  dtype=torch.float32).to(self.args.device).unsqueeze(1)
            set1_x = torch.tensor(np.array([restore_image(set1_x[i]) for i in range(len(set1_x))]),
                                  dtype=torch.float32).to(self.args.device).unsqueeze(1)
            f_set0 = f(set0_x)
            f_set1 = f(set1_x)
            inds = test_dataset.labels[lab].cpu().numpy()
            shapley_value = 1. / M * weights.unsqueeze(dim=0).mm(
                (f_set1[:, inds] + 1e-10).log().reshape(-1, 1)
                - (f_set0[:, inds] + 1e-10).log().reshape(-1, 1))
        return shapley_value

    @torch.no_grad()
    def brute_force_shapley(self, f, x):
        M = x.shape[1]
        shap_index = torch.arange(M)
        mask_dec = torch.arange(0, 2 ** (M-1))
        mask = self.binary(mask_dec, M - 1)

        shapley_value = torch.zeros((1, len(shap_index)))
        for idx, feature_index in enumerate(shap_index):
            shapley_value[:, idx] = self.sub_brute_force_shapley(f, x, mask, feature_index, M).squeeze(dim=0)

        return shapley_value