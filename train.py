import torch
import torch.nn as nn
import torch.optim as optim
from sgd_clip import SGDClipGrad
import copy
from gradient import get_gradient

from sharpness import get_hessian_eigenvalues
def init_corrections(args, model_copies, train_loaders, criterion):
    gradients = [0.0] * args.world_size
    for client in range(args.world_size):
        train_loader = train_loaders[client]
        net = model_copies[client]
        for i in range(args.communication_interval):
            net.zero_grad()
            data = next(iter(train_loader))
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            net.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            current_grad = [
                p.grad.clone().detach() for p in net.parameters() if p.grad is not None
            ]
            if gradients[client] == 0.0:
                gradients[client] = current_grad
            for j in range(len(gradients[client])):
                gradients[client][j] += current_grad[j] / args.communication_interval
    local_corrections = gradients
    global_correction = [0.0] * len(local_corrections[0])
    for client in range(args.world_size):
        for j in range(len(local_corrections[client])):
            global_correction[j] += local_corrections[client][j] / args.world_size
    return local_corrections, global_correction


def train_models(args, model_copies, train_loaders, val_loader, test_loader):
    criterion = nn.CrossEntropyLoss() if args.loss == "cross_entropy" else nn.MSELoss()
    optimizers = [SGDClipGrad(
        model.parameters(),
        lr=args.eta0,
        clipping_param=10**10,
        algorithm=args.algorithm
        ) for model in model_copies]
    # init SCAFFOLD corrections if needed
    if args.algorithm == "scaffold":
# Compute local correction for local clients.
        local_corrections, global_correction = init_corrections(
            args, model_copies, train_loaders, criterion
        )
    else:
        local_corrections = None
        global_correction = None
    
    streams = [torch.cuda.Stream() for _ in range(args.world_size)]
    train_accuracies = [[] for _ in range(args.world_size)]
    train_losses = [[] for _ in range(args.world_size)]
    test_accuracies = []
    test_losses = []

    scaffold_comparisons = []
    local_evals = []
    global_evals = []
    
    for round in range(args.rounds): 
        for client in range(args.world_size):
            stream = streams[client]
            round_avg_grad = [[torch.zeros_like(p.data) for p in model.parameters()] for model in model_copies]
            with torch.cuda.stream(stream):
                model = model_copies[client]
                optimizer = optimizers[client]
                train_loader = train_loaders[client]
                model.train()
                for step in range(args.communication_interval):
                    data = next(iter(train_loader))
                    inputs, labels = data
                    inputs, labels = inputs.cuda(), labels.cuda()
                    model.cuda()

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    round_avg_grad[client] = [
                        round_avg_grad[client][i] + p.grad.data / args.communication_interval
                        for i, p in enumerate(model.parameters())
                    ]
                    optimizer.step(
                        local_correction=local_corrections[client] if local_corrections is not None else None,
                        global_correction=global_correction,
                    )
                    train_losses[client].append(loss.item())
                    with torch.no_grad():
                        if outputs.dim() > 1 and outputs.size(1) > 1:
                            preds = outputs.argmax(dim=1)
                        else:
                            preds = (outputs.squeeze() > 0.5).long()
                        batch_acc = (preds == labels).float().mean().item() * 100.0
                        train_accuracies[client].append(batch_acc)
                    if (round + 1) % (args.rounds // args.num_evals) == 0:
                        local_eval = get_hessian_eigenvalues(model, criterion, [train_loader])
                        local_evals.append(float(local_eval[0]))
                    if args.algorithm == "scaffold" and (round + 1) % (args.rounds // args.num_evals) == 0:
                        torch.cuda.synchronize()
                        
                        for i in range(args.world_size):
                            deepcopy = copy.deepcopy(model_copies[i])
                            local_grad = get_gradient(deepcopy, [train_loaders[i]], criterion)
                            global_grad = get_gradient(deepcopy, train_loaders, criterion)
                            scaffold_grad = local_grad + torch.cat([g.view(-1) for g in global_correction]) - torch.cat([g.view(-1) for g in local_corrections[i]])
                            comparison1 = torch.norm(local_grad - global_grad).item()
                            comparison2 = torch.norm(scaffold_grad - global_grad).item()
                            comparison3 = torch.dot(local_grad, global_grad) / (torch.norm(global_grad) * torch.norm(local_grad) + 1e-10)
                            comparison4 = torch.dot(scaffold_grad, global_grad) / (torch.norm(global_grad) * torch.norm(scaffold_grad) + 1e-10)
                            scaffold_comparisons.append((comparison1, comparison2, comparison3, comparison4))
                local_corrections[client] = round_avg_grad[client]
                global_correction = [0.0] * len(local_corrections[0])
                for i in range(args.world_size):
                    for j in range(len(local_corrections[i])):
                        global_correction[j] += local_corrections[i][j] / args.world_size


        torch.cuda.synchronize()
        # Average model parameters (including buffers) and replace each local model with the averaged model
        avg_state = {}
        # accumulate state_dict tensors on CPU
        for m in model_copies:
            sd = m.state_dict()
            for k, v in sd.items():
                v_cpu = v.detach().cpu()
                if k not in avg_state:
                    avg_state[k] = v_cpu.clone()
                else:
                    avg_state[k] += v_cpu
        # divide to get mean
        for k in avg_state:
            avg_state[k] /= args.world_size
        # load averaged state into each model (on the model's device)
        for m in model_copies:
            device = next(m.parameters()).device
            averaged_sd = {k: avg_state[k].to(device) for k in avg_state}
            m.load_state_dict(averaged_sd)
        # Evaluate models periodically
        if (round + 1) % (args.rounds // args.num_evals) == 0:
            # Evaluate averaged model on test set
            model = model_copies[0]
            model.eval()
            device = next(model.parameters()).device

            total_loss = 0.0
            total_correct = 0
            total_samples = 0

            with torch.no_grad():
                for data in test_loader:
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    batch_size = labels.size(0)
                    total_loss += loss.item() * batch_size

                    if outputs.dim() > 1 and outputs.size(1) > 1:
                        preds = outputs.argmax(dim=1)
                    else:
                        preds = (outputs.squeeze() > 0.5).long()

                    total_correct += (preds == labels).sum().item()
                    total_samples += batch_size

            avg_test_loss = total_loss / total_samples
            test_acc = (total_correct / total_samples) * 100.0

            test_losses.append(avg_test_loss)
            test_accuracies.append(test_acc)

            # Optional: compute global Hessian/eigenvalue estimate on the test set
            global_eval = get_hessian_eigenvalues(model, criterion, train_loaders)
            global_evals.append(float(global_eval[0]))
            # Evaluation code here (omitted for brevity)
    print("Training complete.")
            
    
