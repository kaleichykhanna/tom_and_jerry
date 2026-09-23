import torch

def train_loop(dataloader, model, loss_fn, optimizer, device, scaler=None):
    model.train()
    num_batches = len(dataloader)
    train_loss, train_acc = 0.0, 0.0
    print(f"Training on {num_batches} batches...")

    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device, non_blocking=(device == "cuda"))
        y = y.to(device, non_blocking=(device == "cuda"))

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            enabled=(device.type == "cuda"),
        ):
            pred = model(X)
            loss = loss_fn(pred, y)

        if device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        train_loss += loss.item()
        train_acc += (pred.argmax(1) == y).type(torch.float).mean().item()

        if batch % 100 == 1:
            print(f"Batch {batch}/{num_batches}: Loss: {loss.item():.4f}, Accuracy: {(pred.argmax(1) == y).type(torch.float).mean().item():.4f}")

    train_loss /= num_batches
    train_acc /= num_batches
    print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
    
    return train_loss, train_acc

def test_loop(dataloader, model, loss_fn, device):
    model.eval()
    num_batches = len(dataloader)
    dataset_size = len(dataloader.dataset)
    test_loss, correct = 0, 0
    print(f"Testing on {num_batches} batches...")

    is_cuda = device.type == "cuda"

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device, non_blocking=is_cuda)
            y = y.to(device, non_blocking=is_cuda)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16 if is_cuda else torch.float32,
                enabled=is_cuda,
            ):
                pred = model(X)
                loss = loss_fn(pred, y)
            test_loss += loss.item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    test_acc = correct / dataset_size

    print(f"Test Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")
    return test_loss, test_acc