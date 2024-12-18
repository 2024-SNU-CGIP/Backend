import torch

def train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=50, patience=5):
    best_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for photos_L, photos_U, xrays, labels in train_loader:
            photos_L, photos_U, xrays, labels = photos_L.to(device), photos_U.to(device), xrays.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(photos_L, photos_U, xrays).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')

        val_loss = validate_model(model, val_loader, criterion, device)

        if val_loss < best_loss:
            best_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), 'model_weights.pth')
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f'Early stopping triggered after {epoch + 1} epochs')
            break

    model.load_state_dict(torch.load('model_weights.pth'))

def validate_model(model, data_loader, criterion, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for photos_L, photos_U, xrays, labels in data_loader:
            photos_L, photos_U, xrays, labels = photos_L.to(device), photos_U.to(device), xrays.to(device), labels.to(device)
            outputs = model(photos_L, photos_U, xrays).squeeze()
            val_loss += criterion(outputs, labels).item()
    return val_loss / len(data_loader)

def evaluate_model(model, data_loader, device):
    model.eval()
    test_accuracy = 0

    with torch.no_grad():
        for photos_L, photos_U, xrays, labels in data_loader:
            photos_L, photos_U, xrays, labels = photos_L.to(device), photos_U.to(device), xrays.to(device), labels.to(device)
            outputs = model(photos_L, photos_U, xrays).squeeze()
            test_accuracy += ((outputs > 0.5) == labels).sum().item()
    
    return test_accuracy / len(data_loader.dataset)