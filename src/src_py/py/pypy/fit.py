import torch

def fit (epoch, model, train_dl, test_dl, loss_fn, optimizer):
    """一个epoch的训练"""

    total = 0
    running_loss = 0
    model.train()
    for x, y in train_dl:
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            total += y.size(0)
            running_loss += loss.item()
    epoch_loss = running_loss / len(train_dl)

    test_total = 0
    test_running_loss = 0
    model.eval()
    with torch.no_grad():
        for x, y in test_dl:
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            test_total += y.size(0)
            test_running_loss += loss.item()
    epoch_test_loss = test_running_loss / len(test_dl)

    print('epoch:', epoch,
          'loss:', round(epoch_loss, 8),
          'test_loss', round(epoch_test_loss, 8))
    return epoch_loss, epoch_test_loss