import os
import re
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.utils.prune as prune
import torch_pruning as tp
from torch import nn
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision import datasets, transforms
from torchvision.models import resnet18

from models.fcnn import FCNN
from models.resnet10 import ResNet10
from models.simpleconvnet import SimpleConvNet

# Configurations

# Data

transform_train = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)


training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=transform_train,
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=transform_test,
)

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}


def get_fcnn(num_classes):
    model = FCNN(num_classes=num_classes)
    return model


def get_simpleconvnet(num_classes):
    model = SimpleConvNet(in_channels=1, num_classes=num_classes)
    return model


def get_resnet10(num_classes):
    model = ResNet10(num_classes=num_classes)
    model.conv1 = nn.Conv2d(
        1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )
    return model


def get_resnet18(num_classes):
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(
        1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def show_sample_images(training_data):
    """Display a grid of sample images from the training dataset."""
    fig = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(training_data), size=(1,)).item()
        img, label = training_data[sample_idx]
        fig.add_subplot(rows, cols, i)
        plt.title(labels_map[label])
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()


def get_device(device: str | None = None) -> str:
    if device is None:
        device = (
            torch.accelerator.current_accelerator().type
            if torch.accelerator.is_available()
            else "cpu"
        )
    print(f"Using {device} device")
    return device


def train_loop(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn, device):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )
    return correct, test_loss


def save_model(model, epoch, accuracy, extra=""):
    if extra != "":
        extra = "_" + extra
    filename = os.path.join(
        "results",
        f"{model.__class__.__name__}{extra}_{epoch:05d}_{int(accuracy*10000):05d}.pt",
    )
    # torch.save(model.state_dict(), filename)
    torch.save(model, filename)
    return filename


def export_onnx(model, filename):
    model.eval()
    model.to("cpu")
    input_x = torch.randn(1, 1, 28, 28)
    onnx_filename = filename.replace(".pt", ".onnx")
    torch.onnx.export(
        model,
        input_x,
        onnx_filename,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=12,
        do_constant_folding=True,
        verbose=True,
    )
    print(f"Exported pruned model to {onnx_filename}")
    return filename


def train_model(
    model, epochs, loss_fn, optimizer, scheduler, device, filename_extra=""
):
    tic = time.time()
    accuracy = 0
    filename = ""
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, device)
        new_accuracy, _ = test_loop(test_dataloader, model, loss_fn, device)
        if new_accuracy > accuracy:
            accuracy = new_accuracy
            filename = save_model(model, epoch, accuracy, extra=filename_extra)
        scheduler.step()
    print("Ready")
    toc = time.time()
    print(f"Chrono: {toc - tic:.2f} seconds")
    return filename


def prune_model_structured(
    model,
    epochs,
    loss_fn,
    optimizer,
    scheduler,
    device,
    amount,
    retrain=True,
    export_onnx=True,
):
    model.to(device)
    test_loop(test_dataloader, model, loss_fn, device)
    model.to("cpu")
    model.eval()
    example_inputs = torch.randn(1, 1, 28, 28)
    imp = tp.importance.MagnitudeImportance(p=1)  # L1-norm
    pruner = tp.pruner.MetaPruner(
        model,
        example_inputs=example_inputs,
        importance=imp,
        pruning_ratio=amount,
        iterative_steps=1,
        ignored_layers=[model.conv1, model.fc],
    )
    pruner.step()  # Actually prunes the model
    model.to(device)
    summary(model, input_size=(1, 1, 28, 28), device=device)
    print("Pruned model accuracy:")
    test_loop(test_dataloader, model, loss_fn, device)
    print("Retraining pruned model...")
    if retrain:
        train_model(
            model,
            epochs,
            loss_fn,
            optimizer,
            scheduler,
            device,
            filename_extra=f"structured_pruned_r{int(amount*100):03d}",
        )


def prune_model_structured_buildin(
    model,
    epochs,
    loss_fn,
    optimizer,
    scheduler,
    device,
    amount,
    retrain=True,
    remove_masks=False,
):

    y_max_plot = 100
    weights_before = snapshot_all_conv_weights(model, in_channel=0)
    plot_L1_norm_of_weights_grouped(
        model,
        os.path.join("plots", f"{model.__class__.__name__}_full_L1_norm_plot.jpg"),
        y_max=y_max_plot,
    )

    # Prune
    ignore_names = ["conv1", "fc"]
    for name, module in model.named_modules():
        # Only prune Conv2d layers that are NOT conv1 or fc
        if isinstance(module, torch.nn.Conv2d) and name not in ignore_names:
            prune.ln_structured(module, name="weight", amount=amount, n=1, dim=0)
    model.to(device)
    summary(model, input_size=(1, 1, 28, 28), device=device)

    weights_after = snapshot_all_conv_weights(model, in_channel=0)
    save_all_conv_filters_grid_compare(
        weights_before,
        weights_after,
        out_dir="plots",
        model_name="Model_SP",
        in_channel=0,
    )

    filename = os.path.join(
        "plots", f"{model.__class__.__name__}_structured_pruned_L1_norm_plot.jpg"
    )
    plot_L1_norm_of_weights_grouped(model, filename, y_max=y_max_plot)

    if retrain:
        train_model(
            model,
            epochs,
            loss_fn,
            optimizer,
            scheduler,
            device,
            filename_extra="structured_pruned",
        )
    if remove_masks:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                prune.remove(module, "weight")
    summary(model, input_size=(1, 1, 28, 28), device=device)


def snapshot_all_conv_weights(model, in_channel=0):
    """
    Returns a dict {layer_name: weight_numpy_array} for all Conv2d layers,
    only for the specified in_channel (usually 0).
    """
    weights = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # Shape: (out_channels, in_channels, kH, kW)
            # Only grab weights for a specific input channel for plotting
            weights[name] = module.weight.detach().cpu().numpy()[:, in_channel, :, :]
    return weights


def save_all_conv_filters_grid_compare(
    weights_before, weights_after, out_dir="plots", model_name="Model", in_channel=0
):
    """
    Saves comparison grid plots for all Conv2d layers given weights before and after pruning.
    """
    os.makedirs(out_dir, exist_ok=True)
    for layer_name in weights_before:
        if layer_name in weights_after:
            filename = os.path.join(
                out_dir,
                f"{model_name}_{layer_name.replace('.', '_')}_in{in_channel}.jpg",
            )
            save_conv_filters_grid_compare(
                weights_before[layer_name],
                weights_after[layer_name],
                layer_name=layer_name,
                in_channel=in_channel,
                filename=filename,
            )


def prune_model_unstructured(
    model, epochs, loss_fn, optimizer, scheduler, device, amount, retrain=True
):
    y_max_plot = 100
    weights_before = snapshot_all_conv_weights(model, in_channel=0)
    plot_L1_norm_of_weights_grouped(
        model,
        os.path.join("plots", f"{model.__class__.__name__}_full_L1_norm_plot.jpg"),
        y_max=y_max_plot,
    )

    # Prune
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            parameters_to_prune.append((module, "weight"))
    prune.global_unstructured(
        parameters_to_prune, pruning_method=prune.L1Unstructured, amount=amount
    )
    model.to(device)

    weights_after = snapshot_all_conv_weights(model, in_channel=0)
    save_all_conv_filters_grid_compare(
        weights_before, weights_after, out_dir="plots", model_name="Model", in_channel=0
    )

    filename = os.path.join(
        "plots", f"{model.__class__.__name__}_pruned_L1_norm_plot.jpg"
    )
    plot_L1_norm_of_weights_grouped(model, filename, y_max=y_max_plot)
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            w = module.weight
            perc = 100 * (w == 0).sum().item() / w.numel()
            if perc > 90:
                print(
                    f"Layer {name} is over 90% pruned! Consider removing or reducing this layer."
                )

    if retrain:
        train_model(
            model,
            epochs,
            loss_fn,
            optimizer,
            scheduler,
            device,
            filename_extra="unstructured_pruned",
        )
    return model


def get_layer_weights(model, layer_name, in_channel=0):
    """Extract (out_channels, kH, kW) weights for a given Conv2d layer and input channel."""
    layer = dict(model.named_modules())[layer_name]
    # PyTorch Conv2d weights: (out_channels, in_channels, kH, kW)
    weights = layer.weight.detach().cpu().numpy()[:, in_channel, :, :]
    return weights


def visualize_conv_filters_grid(weights, title="", vmax=None, vmin=None):
    """Plot all filters as small squares in a grid."""
    out_channels, kH, kW = weights.shape
    grid_size = int(np.ceil(np.sqrt(out_channels)))
    fig, axes = plt.subplots(
        grid_size, grid_size, figsize=(grid_size * 2, grid_size * 2)
    )
    vmax = vmax if vmax is not None else np.abs(weights).max()
    vmin = vmin if vmin is not None else -vmax
    for i, ax in enumerate(axes.flat):
        if i < out_channels:
            filt = weights[i]
            ax.imshow(filt, cmap="bwr", vmin=vmin, vmax=vmax)
        ax.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def save_conv_filters_grid_compare(
    weights_before, weights_after, layer_name="conv", in_channel=0, filename=None
):
    """Compare before/after pruning for all filters and save to file."""
    vmax = max(np.abs(weights_before).max(), np.abs(weights_after).max())
    vmin = -vmax
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    for ax, weights, title in zip(
        axs,
        [weights_before, weights_after],
        [
            f"{layer_name} in_ch={in_channel}\nBefore pruning",
            f"{layer_name} in_ch={in_channel}\nAfter pruning",
        ],
    ):
        out_channels, kH, kW = weights.shape
        grid_size = int(np.ceil(np.sqrt(out_channels)))
        # Compose a large blank image for the grid
        big_img = np.zeros((grid_size * kH, grid_size * kW))
        for idx in range(out_channels):
            row = idx // grid_size
            col = idx % grid_size
            big_img[row * kH : (row + 1) * kH, col * kW : (col + 1) * kW] = weights[idx]
        ax.imshow(big_img, cmap="bwr", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    # filename = f"{filename_prefix}_{layer_name.replace('.', '_')}_in{in_channel}.png"

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename, bbox_inches="tight")
    plt.close(fig)  # Close the figure to avoid display
    print(f"Saved filter grid comparison to {filename}")


def plot_activation(model, train_loader, filename):
    activations = {}

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach().cpu().abs().mean().item()

        return hook

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            module.register_forward_hook(get_activation(name))

    # Run a forward pass
    model.eval()
    with torch.no_grad():
        for images, _ in train_loader:
            model(images.to(next(model.parameters()).device))
            break  # Just one batch for visualization
    plt.figure(figsize=(10, 5))
    plt.bar(activations.keys(), activations.values())
    plt.ylabel("Mean Activation")
    plt.title("Mean Activation per Layer")
    plt.xticks(rotation=45)
    plt.savefig(filename)


def plot_all_conv_filters(
    model,
    layer_name="conv1",
    in_channel=0,
    before_prune=None,
    after_prune=None,
    title_prefix="",
    filename=None,
):
    """
    Show all filters (output channels) of a convolutional layer for a given input channel,
    side by side with thin separators, before and after pruning.
    """

    # Helper to extract weights
    def extract_weights(state, layer_name, in_channel):
        w = state[layer_name]  # shape: (out_ch, in_ch, kH, kW)
        return w[:, in_channel, :, :]  # shape: (out_ch, kH, kW)

    # Get weights before and after pruning
    if before_prune is not None and after_prune is not None:
        w_before = extract_weights(before_prune, layer_name, in_channel)
        w_after = extract_weights(after_prune, layer_name, in_channel)
    else:
        module = dict(model.named_modules())[layer_name]
        w_before = module.weight.detach().cpu().numpy()[:, in_channel, :, :]
        w_after = w_before  # No after_prune given; use current weights

    num_filters, kH, kW = w_before.shape
    sep = 1  # 1-pixel separator

    # Prepare canvas for before and after
    def build_canvas(weights):
        # Each filter goes side-by-side, separated by 1-pixel line
        canvas = (
            np.ones((kH, num_filters * kW + (num_filters - 1) * sep)) * np.nan
        )  # nan for gray separator
        for i in range(num_filters):
            start = i * (kW + sep)
            canvas[:, start : start + kW] = weights[i]
        return canvas

    canvas_before = build_canvas(w_before)
    canvas_after = build_canvas(w_after)

    vmin = min(np.nanmin(canvas_before), np.nanmin(canvas_after))
    vmax = max(np.nanmax(canvas_before), np.nanmax(canvas_after))

    fig, axs = plt.subplots(1, 2, figsize=(20, 3))
    axs[0].imshow(canvas_before, aspect="auto", cmap="bwr", vmin=vmin, vmax=vmax)
    axs[0].set_title(f"{title_prefix}{layer_name} (before pruning)")
    axs[1].imshow(canvas_after, aspect="auto", cmap="bwr", vmin=vmin, vmax=vmax)
    axs[1].set_title(f"{title_prefix}{layer_name} (after pruning)")
    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)


def plot_conv_weights(
    model,
    layer_name="conv1",
    out_channel=0,
    in_channel=0,
    before_prune=None,
    after_prune=None,
    filename=None,
):
    """
    Visualize the weights of a single filter (out_channel, in_channel) before and after pruning.
    If before_prune/after_prune are given, use those instead of current model.
    """
    # Fetch the weights
    module = dict(model.named_modules())[layer_name]
    if before_prune is not None:
        w_before = before_prune[layer_name][out_channel, in_channel, :, :]
    else:
        w_before = module.weight.detach().cpu().numpy()[out_channel, in_channel, :, :]
    if after_prune is not None:
        w_after = after_prune[layer_name][out_channel, in_channel, :, :]
    else:
        w_after = module.weight.detach().cpu().numpy()[out_channel, in_channel, :, :]

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(w_before, cmap="bwr", interpolation="nearest")
    axs[0].set_title("Before Pruning")
    axs[1].imshow(w_after, cmap="bwr", interpolation="nearest")
    axs[1].set_title("After Pruning")
    plt.suptitle(f"{layer_name} weights: out_ch {out_channel}, in_ch {in_channel}")
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)


def snapshot_conv_weights(model):
    """Return a dict of conv layer weights for later comparison."""
    state = {}
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            state[name] = m.weight.detach().cpu().numpy().copy()
    return state


def plot_L1_norm_of_weights_grouped(model, filename: str, y_max: int | None = None):
    """
    Plots the L1 norm of weights per output channel for each Conv2d layer,
    grouping by blocks (e.g. layer1, layer2, etc). All subplots share the same y-scale.
    """
    # Collect all conv layers and group them by block (using name pattern)
    convs = []
    names = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            convs.append(module)
            names.append(name)
    # Determine block groupings from layer names (e.g., 'layer1', 'layer2', etc.)
    block_groups = {}
    for idx, name in enumerate(names):
        m = re.search(r"(layer\d+)", name)
        block = m.group(1) if m else "other"
        block_groups.setdefault(block, []).append((name, convs[idx]))
    # For subplot layout
    num_blocks = len(block_groups)
    fig, axes = plt.subplots(1, num_blocks, figsize=(10 * num_blocks, 10), sharey=True)
    if num_blocks == 1:
        axes = [axes]
    # Compute global min/max for y-axis scale
    all_norms = []
    for blocks in block_groups.values():
        for name, conv in blocks:
            w = conv.weight.detach().cpu().numpy()
            l1_norms = np.abs(w).sum(axis=(1, 2, 3))
            all_norms.append(l1_norms)
    y_min = 0
    if y_max is None:
        y_max = max([arr.max() for arr in all_norms])
    # Plot each block group as a subplot
    colors = ["blue", "red", "gray", "orange", "purple"]
    for ax, (block, convs_in_block) in zip(axes, block_groups.items()):
        i = 0
        for name, conv in convs_in_block:
            w = conv.weight.detach().cpu().numpy()
            l1_norms = np.abs(w).sum(axis=(1, 2, 3))
            ax.bar(
                np.arange(len(l1_norms)),
                l1_norms,
                label=name,
                color=colors[i],
                alpha=0.5,
            )
            i += 1
        ax.set_title(f"L1 Norm per Channel - {block}")
        ax.set_xlabel("Output Channel")
        ax.set_ylabel("L1 Norm")
        ax.set_ylim([y_min, y_max * 1.1])
        ax.legend()
    plt.tight_layout()
    plt.savefig(filename)


def get_model(model_name, num_classes):
    if model_name == "fcnn":
        return get_fcnn(num_classes)
    elif model_name == "simpleconvnet":
        return get_simpleconvnet(num_classes)
    elif model_name == "resnet10":
        return get_resnet10(num_classes)
    elif model_name == "resnet18":
        return get_resnet18(num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return model


if __name__ == "__main__":
    BATCH_SIZE = 64
    NUM_CLASSES = 10
    EPOCHS = 15
    LEARNING_RATE = 1.0e-3
    MODEL_NAME = "resnet10"

    TRAIN = True
    PRUNE = False
    PRUNE_STRUCTURED_BUILDIN = False
    PRUNE_STRUCTURED = False
    output_filename = "./results/ResNet10_00013_09265.pt"

    os.makedirs("data", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    device = get_device()

    if TRAIN:
        model = get_model(MODEL_NAME, num_classes=NUM_CLASSES)
        summary(model, input_size=(1, 1, 28, 28), device=device)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        output_filename = train_model(
            model, EPOCHS, loss_fn, optimizer, scheduler, device
        )

    model = torch.load(output_filename, map_location=device, weights_only=False)
    export_onnx(model, output_filename)

    if PRUNE:
        model = torch.load(output_filename, map_location=device, weights_only=False)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE * 0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        pruned_model = prune_model_unstructured(
            model,
            EPOCHS // 3,
            loss_fn,
            optimizer,
            scheduler,
            device,
            amount=0.9,
            retrain=False,
        )

    if PRUNE_STRUCTURED_BUILDIN:

        model = torch.load(output_filename, map_location=device, weights_only=False)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE * 0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        prune_model_structured_buildin(
            model,
            EPOCHS,
            loss_fn,
            optimizer,
            scheduler,
            device,
            amount=0.5,
            retrain=False,
        )

    if PRUNE_STRUCTURED:

        for i in range(1, 10):
            amount = i / 10

            # model = get_model(MODEL_NAME, num_classes=NUM_CLASSES)
            # model.load_state_dict(torch.load(output_filename, map_location=device))
            model = torch.load(output_filename, map_location=device, weights_only=False)

            loss_fn = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE * 0.1)
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=5, gamma=0.5
            )
            prune_model_structured(
                model,
                EPOCHS,
                loss_fn,
                optimizer,
                scheduler,
                device,
                amount=amount,
                retrain=True,
            )
