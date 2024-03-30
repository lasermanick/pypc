import os
import pprint
import sys

import torch

from tqdm import tqdm

from pypc import utils
from pypc import datasets
from pypc import optim
from pypc.models import PCModel

import neptune
from pypc import constants  # Defines secret API_KEY_NEPTUNE for Neptune access

from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_example_data_MNIST(dataloader, batch_num=0, path=None, cmap="gray"):
    if batch_num >= len(dataloader):
        batch_num = -1
    image_batch = dataloader[batch_num][0]
    label_batch = dataloader[batch_num][1]
    images = image_batch.cpu().detach().numpy()
    labels = label_batch.cpu().detach().numpy()
    _, indices = np.unique(labels, axis=0, return_index=True)
    indices = np.flip(indices)
    # HACK to plot first ten images (not first examples of digits 0 to 9)
    # indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    fig, axes = plt.subplots(2, 5)
    fig.set_size_inches(8, 3)
    fig.set_dpi(150)
    axes = axes.flatten()
    plt.setp(axes, xticks=[0, 27])
    plt.setp(axes, yticks=[0, 27])
    for i in range(10):
        axes[i].tick_params(top=False, labeltop=False, bottom=False, labelbottom=False, width=2)
        axes[i].tick_params(left=False, labelleft=False, right=False, labelright=False, width=2)
        axes[i].imshow(images[indices[i]].reshape(28, 28), cmap=cmap)
    axes[0].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False, labelsize=16)
    axes[0].tick_params(left=True, labelleft=True, right=False, labelright=False, labelsize=16)

    if path:
        plt.savefig(path)
    plt.show()
    plt.close("all")

def calc_accuracy(data_loader, model, config, file_prefix="", log_test=False):
    acc = 0
    for batch_id, (img_batch, label_batch) in enumerate(tqdm(data_loader, file=sys.stdout)):
        # model.set_precisions_by_per_pixel_variance(img_batch)
        label_preds = model.test_batch_generative(
            img_batch, config.n_test_iters, init_std=config.init_std, fixed_preds=config.fixed_preds_test, log_batch=(log_test and (batch_id == len(data_loader)-1))
        )
        acc += datasets.accuracy(label_preds, label_batch)
        datasets.save_csv(label_preds, cf.preddir + "preds_" + f"{file_prefix}b{batch_id:03}.csv")
        datasets.save_csv(label_batch, cf.preddir + "actuals_" + f"{file_prefix}b{batch_id:03}.csv")
    return acc / len(data_loader)

def print_batch_stats(img_batch, cf):
    first_noisy_pixel = int(img_batch.size(1) * (1.0 - cf.noise_coverage))
    last_noisy_pixel = img_batch.size(1)
    print(f"Noise coverage = {cf.noise_coverage:.2f}")
    print(f"First noisy pixel = {first_noisy_pixel}")
    print(f"Last noisy pixel = {last_noisy_pixel}")
    print(f"Full image")
    print(f"Batch size = {img_batch.size()}")
    print(f"Batch mean = {img_batch.mean():.4f}")
    print(f"Batch variance = {img_batch.var():.4f}")
    print(f"Batch stddev = {img_batch.std():.4f}")
    print(f"Region without added noise")
    print(f"Batch size = {img_batch[:, 0:first_noisy_pixel].size()}")
    print(f"Batch mean = {img_batch[:, 0:first_noisy_pixel].mean():.4f}")
    print(f"Batch variance = {img_batch[:, 0:first_noisy_pixel].var():.4f}")
    print(f"Batch stddev = {img_batch[:, 0:first_noisy_pixel].std():.4f}")
    print(f"Region with added noise")
    print(f"Batch size = {img_batch[:, first_noisy_pixel:last_noisy_pixel].size()}")
    print(f"Batch mean = {img_batch[:, first_noisy_pixel:last_noisy_pixel].mean():.4f}")
    print(f"Batch variance = {img_batch[:, first_noisy_pixel:last_noisy_pixel].var():.4f}")
    print(f"Batch stddev = {img_batch[:, first_noisy_pixel:last_noisy_pixel].std():.4f}")

def main(cf):
    print(f"\nStarting generative experiment {cf.logdir}: --seed {cf.seed} --device {utils.DEVICE}")
    print(datetime.now())
    os.makedirs(cf.logdir, exist_ok=True)
    os.makedirs(cf.imgdir, exist_ok=True)
    os.makedirs(cf.preddir, exist_ok=True)
    utils.seed(cf.seed)

    pprint.pprint(cf)
    if cf.json_enabled:
        utils.save_json({k: str(v) for (k, v) in cf.items()}, cf.logdir + "config.json")
    if cf.neptune_enabled:
        cf.run_log["parameters"] = cf  # Log configuration to Neptune

    # Use per pixel scaling of noise (and precision)
    # noise_per_pixel_scaling = torch.rand((28, 28)) * (cf.noise_max_std - cf.noise_min_std) + cf.noise_min_std
    # precis_per_pixel = 1 / (1 + noise_per_pixel_scaling ** 2.0)
    noise_per_pixel_scaling = None
    precis_per_pixel = None

    print("Loading data...")
    # train_dataset = datasets.MNIST(train=True, scale=cf.label_scale, size=cf.train_size, normalize=cf.normalize, add_noise=cf.add_noise, noise_mean=cf.noise_mean, noise_std=cf.noise_std, noise_coverage=cf.noise_coverage, noise_per_pixel_scaling=noise_per_pixel_scaling)
    # test_dataset = datasets.MNIST(train=False, scale=cf.label_scale, size=cf.test_size, normalize=cf.normalize, add_noise=cf.add_noise, noise_mean=cf.noise_mean, noise_std=cf.noise_std, noise_coverage=cf.noise_coverage, noise_per_pixel_scaling=noise_per_pixel_scaling)
    train_dataset = datasets.FashionMNIST(train=True, scale=cf.label_scale, size=cf.train_size, normalize=cf.normalize, add_noise=cf.add_noise, noise_mean=cf.noise_mean, noise_std=cf.noise_std, noise_coverage=cf.noise_coverage, noise_per_pixel_scaling=noise_per_pixel_scaling)
    test_dataset = datasets.FashionMNIST(train=False, scale=cf.label_scale, size=cf.test_size, normalize=cf.normalize, add_noise=cf.add_noise, noise_mean=cf.noise_mean, noise_std=cf.noise_std, noise_coverage=cf.noise_coverage, noise_per_pixel_scaling=noise_per_pixel_scaling)
    train_loader = datasets.get_dataloader(train_dataset, cf.batch_size)
    test_loader = datasets.get_dataloader(test_dataset, cf.batch_size)
    print(f"Loaded data [training batches: {len(train_loader)}, test batches: {len(test_loader)}]")

    plot_example_data_MNIST(train_loader, batch_num=0, path=cf.imgdir + "example_training_data.png", cmap='rainbow')
    # HACK to plot test data (see also the function)
    # plot_example_data_MNIST(test_loader, batch_num=0, path=cf.imgdir + "example_training_data.png", cmap='rainbow')

    model = PCModel(
        nodes=cf.nodes,
        mu_dt=cf.mu_dt,
        act_fn=cf.act_fn,
        use_bias=cf.use_bias,
        kaiming_init=cf.kaiming_init,
        use_precis=cf.use_precis,
        precis_factor=cf.precis_factor,
        precis_coverage=cf.precis_coverage,
        precis_per_pixel=precis_per_pixel,
        run_log=cf.run_log,
        log_node_its=cf.log_node_its,
    )
    optimizer = optim.get_optim(
        model.params,
        cf.optim,
        cf.lr,
        batch_scale=cf.batch_scale,
        grad_clip=cf.grad_clip,
        weight_decay=cf.weight_decay,
    )

    with torch.no_grad():  # Disable automatic gradient calculation
        metrics = {"test_acc": [], "train_acc": []}
        for epoch in range(1, cf.n_epochs + 1):
            print(f"\nTrain @ epoch {epoch} ({len(train_loader)} batches)")

            # Train each batch
            for batch_id, (img_batch, label_batch) in enumerate(tqdm(train_loader, file=sys.stdout)):
                # print_batch_stats(img_batch, cf)
                # model.set_precisions_by_per_pixel_variance(img_batch)
                model.train_batch_generative(
                    img_batch, label_batch, cf.n_train_iters, fixed_preds=cf.fixed_preds_train, log_batch=(batch_id == len(train_loader)-1)
                )
                optimizer.step(
                    curr_epoch=epoch,
                    curr_batch=batch_id,
                    n_batches=len(train_loader),
                    batch_size=img_batch.size(0),
                )

            # Evaluate the model at specified intervals
            if epoch % cf.eval_every == 0:
                print(f"\nEvaluate @ epoch {epoch}")

                # Calculate training accuracy
                if cf.eval_train_acc:
                    acc = calc_accuracy(train_loader, model, cf, f"train_e{epoch:03}_", log_test=False)
                    print(f"Train accuracy: {acc:.4f}")
                    metrics["train_acc"].append(acc)
                    if cf.neptune_enabled:
                        cf.run_log["train/acc"].log(acc)

                # Calculate test accuracy
                if cf.eval_test_acc:
                    acc = calc_accuracy(test_loader, model, cf, f"test_e{epoch:03}_", log_test=True)
                    print(f"Test accuracy: {acc:.4f}")
                    metrics["test_acc"].append(acc)
                    if cf.neptune_enabled:
                        cf.run_log["test/acc"].log(acc)

                # Generate image predictions
                if cf.eval_images:
                    # _, label_batch = next(iter(test_loader))  # Use first batch
                    label_batch = utils.set_tensor(torch.diagflat(torch.ones(10)))  # Use each of the one hot encoded labels
                    img_preds = model.forward(label_batch)
                    datasets.plot_imgs_alt(img_preds, path=cf.imgdir + f"e{epoch:03}.png", cmap="rainbow")

                    # preds = pd.read_csv(r'D:\n\OneDrive\Documents\GitHub\pypc-desktop\scripts\data\230727_101349 - copy for analysis\000\preds\preds_test_e015_b000.csv', nrows=10, index_col=0)
                    # label_batch = utils.set_tensor(torch.tensor(preds.to_numpy()))
                    # img_preds = model.forward(label_batch)
                    # datasets.plot_imgs_alt(img_preds, path=cf.imgdir + f"alt e{epoch:03}.png", cmap="rainbow")

                # Save metrics to json log
                if cf.json_enabled:
                    utils.save_json(metrics, cf.logdir + "metrics.json")

    if cf.neptune_enabled:
        cf.run_log.stop()  # Stop logging to Neptune.ai


# INSTRUCTIONS
# For a single point calculation, set scan_1d and scan_2d to [0] and ignore s1d and s2d
# For a 1D scan, set scan_1d to a list of parameter values, set scan_2d to [0], use s1d and ignore s2d
# For a 2D scan, set scan_1d and scan_2d to lists of parameter values, use s1d and s2d
# In all cases, use cf.seeds to repeat runs (and, for easier analysis, gather into a Neptune group if used)
if __name__ == "__main__":
    cf = utils.AttrDict()  # Create configuration
    # scan_1d = [0, 0.002, 0.005, 0.01, 0.02, 0.027, 0.038, 0.059, 0.1, 0.2, 0.5, 0.8, 1.0, 1.2]  # List of parameters for 1D parameter scan
    scan_1d = [0]  # List of parameters for 1D parameter scan
    scan_2d = [0]  # List of parameters for 2D parameter scan
    for s2d in scan_2d:
        for s1d in scan_1d:
            cf.seeds = [0]  # Create list of >=1 seeds for repeat runs
            now = datetime.now().strftime('%y%m%d_%H%M%S')
            cf.run_group = ""
            for seed in cf.seeds:
                # logging params
                cf.log_node_its = False  # WARNING: Can log A LOT of data to Neptune if True
                cf.json_enabled = True
                cf.neptune_enabled = False
                cf.neptune_mode = "async"  # https://docs.neptune.ai/api/connection_modes/
                cf.neptune_project = "lasermanick/PYPC"
                cf.neptune_api_token = constants.API_KEY_NEPTUNE
                if cf.neptune_enabled:
                    cf.run_log = neptune.init_run(mode=cf.neptune_mode, project=cf.neptune_project, api_token=cf.neptune_api_token)
                    run_id = cf.run_log["sys/id"].fetch()
                    if cf.run_group == "":
                        cf.run_group = "grp_" + run_id
                    cf.logdir = f"data/generative/{now}_{cf.run_group}/{seed:03}_{run_id}/"
                else:
                    cf.run_log = None
                    cf.logdir = f"data/generative/{now}/{seed:03}/"
                cf.imgdir = cf.logdir + "imgs/"
                cf.preddir = cf.logdir + "preds/"

                # experiment params
                cf.seed = seed
                cf.n_epochs = 20  # 20
                cf.eval_every = 1
                cf.eval_train_acc = False
                cf.eval_test_acc = True
                cf.eval_images = True

                # model params
                cf.use_bias = True
                cf.kaiming_init = False
                cf.nodes = [10, 144, 169, 784]  # [10, 100, 300, 784]
                cf.act_fn = utils.Tanh()
                cf.use_precis = True
                cf.precis_factor = [1.0, 1.0, 1.0, 0.1]  # Errors and precisions at layer 0 are not used
                cf.precis_coverage = [0.0, 0.0, 0.0, 0.5]  # Errors and precisions at layer 0 are not used

                # dataset params
                cf.train_size = None  # None
                cf.test_size = None  # None
                cf.label_scale = None  # None
                cf.normalize = True  # None
                cf.add_noise = True  # False
                cf.noise_mean = 0.0
                cf.noise_std = 2.0
                cf.noise_coverage = 0.5
                # cf.noise_min_std = 3.0
                # cf.noise_max_std = 3.0

                # optim params
                cf.optim = "Adam"  # "Adam"
                cf.lr = 0.006  # 0.006
                cf.batch_size = 10000  # 10000
                cf.batch_scale = False
                cf.grad_clip = None
                cf.weight_decay = 0.01  # None

                # inference params
                cf.mu_dt = 0.01  # 0.01
                cf.n_train_iters = 50  # 50
                # cf.n_test_iters = min(round(200 / (s1d + 1e-9)), 4000)  # 200
                cf.n_test_iters = 200
                cf.init_std = 0.01
                cf.fixed_preds_train = False
                cf.fixed_preds_test = False

                main(cf)
print("Finishing experiment")
print(datetime.now())
