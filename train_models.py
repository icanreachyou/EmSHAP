from model import *
from masking import *
from early_stop import *

def train_proposal_model(args, loader):
    proposal_network = Gru_proposal_network(args)
    proposal_network.train()
    proposal_network.cuda()
    his_loss = []
    crition = torch.nn.MSELoss()
    optimizer_proposal = torch.optim.Adam(proposal_network.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    early_stopping1 = EarlyStopping('./model/early_proposal/', patience=args.patience)
    for proposal_epoch in range(args.epochs):
        total_loss_proposal = 0
        for batch_idx, data in enumerate(loader):
            optimizer_proposal.zero_grad()
            if args.masking_type == 'dmg':
                dmg = DynamicMaskGenerator(proposal_epoch,
                                           args.epochs,
                                           args.masking_rate_min,
                                           args.masking_rate_max
                                           )
            if args.masking_type == 'uniform':
                dmg = UniformMaskGenerator()
            if args.masking_type == 'bernoulli':
                dmg = BernoulliMaskGenerator(p=args.masking_rate_bernoulli)
            mask_ = []
            for i in range(data[0].shape[0]):
                mask_.append(dmg((args.num_seq, args.num_features)))
            mask_ = np.array(mask_).reshape(data[0].shape)
            data = data[0].cuda().float()
            if len(data.shape) == 2:
                data = data.unsqueeze(1)
                mask_ = mask_[:, np.newaxis, : ]
            proposal_dis, context = proposal_network(data, torch.Tensor(mask_).cuda())
            samples = proposal_dis.sample((args.sample_num,))
            mean_sample = samples.mean(dim=0)
            proposal_mean = torch.reshape(mean_sample, [-1, args.num_seq, args.num_features]) * torch.Tensor(1-mask_).cuda()
            loss_proposal = (-proposal_dis.log_prob(data).sum() +
                             crition(proposal_mean + data * (torch.Tensor(mask_).cuda()), data))
            loss_proposal.backward()
            total_loss_proposal = total_loss_proposal + loss_proposal.item()
            optimizer_proposal.step()

        if proposal_epoch % 1 == 0:
            print('Epoch: {:d}, train loss: {:.3f}'.format(proposal_epoch, total_loss_proposal/args.batch_size))

        his_loss.append(total_loss_proposal/args.batch_size)

        early_stopping1(total_loss_proposal/args.batch_size, proposal_network)

        if early_stopping1.early_stop:
            print("Training stopped early.")
            break
    torch.save(proposal_network.state_dict(),
               './model' + '/' + '{}_proposal_network_{}.pth'.format(args.proposal_network, args.data))
    return proposal_network

def train_energy_model(args, loader, proposal_network):
    energy_network = Energy_network(args)
    energy_network.train()
    energy_network.cuda()
    his_loss = []
    optimizer_energy = torch.optim.Adam(energy_network.parameters(), lr=0.001, weight_decay=5e-3)
    crition = torch.nn.MSELoss()
    
    for energy_epoch in range(args.epochs):
        total_loss_energy = 0
        for batch_idx, data in enumerate(loader):
            optimizer_energy.zero_grad()

            data = data[0].cuda().float()  # [B, seq, F]
            data = data.unsqueeze(1)  # [B, seq, F]

            if args.masking_type == 'dmg':
                dmg = DynamicMaskGenerator(energy_epoch,
                                           args.epochs,
                                           args.masking_rate_min,
                                           args.masking_rate_max
                                           )
            if args.masking_type == 'uniform':
                dmg = UniformMaskGenerator()
            if args.masking_type == 'bernoulli':

                dmg = BernoulliMaskGenerator(p=args.masking_rate_bernoulli)
            mask_ = []
            for i in range(data.shape[0]):
                mask_.append(dmg((args.num_seq, args.num_features)))
            mask_ = np.array(mask_).reshape(data.shape)
            mask_ = torch.tensor(mask_, device=data.device, dtype=torch.float32)
            mask_ = mask_# [B, seq, F]

            query = 1 - mask_
            x_u = data * query

            with torch.no_grad():
                proposal_dist_detach, context = proposal_network(data, mask_)  # context: [B, ctx]

            if args.use_proposal_mean:
                proposal_samples = proposal_dist_detach.mean.unsqueeze(1) # [B,1,seq,F]
                proposal_samples_proposal_ll = proposal_dist_detach.log_prob(proposal_samples)
            else:
                proposal_samples = proposal_dist_detach.sample((args.sample_num,)).unsqueeze(1)  # [K,B,1,seq,F]
                proposal_samples = proposal_samples.permute(1, 0, 2, 3, 4)  # [B,K,1,seq,F]
                proposal_samples_proposal_ll = proposal_dist_detach.log_prob(proposal_samples)

            x_u_and_samples = torch.cat([x_u.unsqueeze(0).unsqueeze(0), proposal_samples], dim=1)  # [B,1+K,1,seq,F]

            B, K_plus1, bat, S, F = x_u_and_samples.shape
            x_u_i = x_u_and_samples.reshape(-1)  # flatten
            u_i = torch.arange(F, device=data.device).repeat(K_plus1, bat, S, 1).reshape(-1)

            tiled_context = context.repeat(1, K_plus1, 1, 1, 1)  # [B,K+1,...]
            tiled_context = tiled_context.reshape(-1, args.context_units)

            negative_energies = energy_network(x_u_i, u_i, tiled_context)
            negative_energies = negative_energies.view(B, K_plus1, bat, S, F) * query

            unnorm_energy_ll = negative_energies[:, 0]  # [B,S,F]
            proposal_samples_unnorm_energy_ll = negative_energies[:, 1:]  # [B,K,S,F]

            proposal_samples_log_ratios = proposal_samples_unnorm_energy_ll - proposal_samples_proposal_ll

            log_normalizers = (torch.logsumexp(proposal_samples_log_ratios, dim=1)
                               - torch.log(
                        torch.tensor(args.sample_num, device=data.device, dtype=torch.float32))) * query

            is_weights = torch.softmax(proposal_samples_log_ratios, dim=1)  # [B,K,S,F]
            energy_mean = (is_weights * proposal_samples).sum(dim=1) * query

            energy_ll = unnorm_energy_ll - log_normalizers
            loss_energy = -energy_ll.sum() + crition(
                energy_mean.squeeze(0) + data* mask_, data)

            loss_energy.backward()
            total_loss_energy += loss_energy.item()
            optimizer_energy.step()

        print('Epoch: {:d}, train loss: {:.3f}'.format(energy_epoch, total_loss_energy / args.batch_size))
        his_loss.append(total_loss_energy / args.batch_size)

    return energy_network
