import torch
def get_gradient(model, train_loaders, criterion):
    grads = 0.0
    n = 0
    for train_loader in train_loaders:
        model.train()
        inputs, labels = next(iter(train_loader))
        inputs, labels = inputs.cuda(), labels.cuda()
        model.cuda()

        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        grad = torch.nn.utils.parameters_to_vector([param.grad for param in model.parameters()]).detach().cpu()
        grads = grads * n / (n + 1) + grad / (n + 1)
        n += 1
    return grads