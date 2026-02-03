import torch
def get_gradient(model, train_loaders, criterion):
    grads = 0.0
    loss_total = 0
    n = 0
    for train_loader in train_loaders:
        model.train()
        inputs, labels = train_loader[0]

        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        loss_total += float(loss.item())
        grad = torch.nn.utils.parameters_to_vector([param.grad for param in model.parameters()]).detach().clone()
        grads = grads * n / (n + 1) + grad / (n + 1)
        n += 1
    return grads, loss / n
