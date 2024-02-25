import torch

class Utils:
    def train_fn(self, model, loss_fn, dataloader, optimizer, device, clip_size=1.0) -> float:
        model.train()
        current_loss = 0.0
        
        for batch, (source, target) in enumerate(dataloader):
            source = source.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(source, target)
            output = output[1:].reshape(-1, output.shape[-1])
            target = target[1:].reshape(-1)
            loss = loss_fn(output, target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_size)
            optimizer.step()

            current_loss += loss.item()
        avg_loss = current_loss / len(dataloader)
        return avg_loss

    def evaluate_fn(self, model, loss_fn, dataloader, device) -> float:
        model.eval()
        current_loss = 0.0
        
        with torch.no_grad():
            for batch, (source, target) in enumerate(dataloader):
                source = source.to(device)
                target = target.to(device)

                output = model(source, target)
                output = output[1:].reshape(-1, output.shape[-1])
                target = target[1:].reshape(-1)
                loss = loss_fn(output, target)

                current_loss += loss.item()

        avg_loss = current_loss / len(dataloader)
        return avg_loss