import os
import pprint

import torch
import torchvision
import matplotlib.pyplot as plt

from tqdm import tqdm
from time import sleep

from pypc import utils
from pypc import datasets
from pypc import optim
from pypc.models import PCModel

import neptune
from pypc import constants  # Defines secret API_KEY_NEPTUNE for Neptune access

from torch.utils.tensorboard import SummaryWriter

# helper function to show an image
# (used in the `plot_classes_preds` function below)
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.cpu().numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

def main(cf):
    print(f"\nStarting supervised experiment {cf.logdir}: --seed {cf.seed} --device {utils.DEVICE}")
    pprint.pprint(cf)
    os.makedirs(cf.logdir, exist_ok=True)
    utils.seed(cf.seed)
    utils.save_json({k: str(v) for (k, v) in cf.items()}, cf.logdir + "config.json")

    print("Loading data...")
    train_dataset = datasets.MNIST(train=True, scale=cf.label_scale, size=cf.train_size, normalize=cf.normalize)
    test_dataset = datasets.MNIST(train=False, scale=cf.label_scale, size=cf.test_size, normalize=cf.normalize)
    train_loader = datasets.get_dataloader(train_dataset, cf.batch_size)
    test_loader = datasets.get_dataloader(test_dataset, cf.batch_size)
    print(f"Loaded data [train batches: {len(train_loader)} test batches: {len(test_loader)}]")

    run = neptune.init(
        project="lasermanick/PYPC",
        api_token=constants.API_KEY_NEPTUNE,
    )
    model = PCModel(
        nodes=cf.nodes,
        mu_dt=cf.mu_dt,
        act_fn=cf.act_fn,
        use_bias=cf.use_bias,
        kaiming_init=cf.kaiming_init,
        use_precis=cf.use_precis,
        precis=cf.precis,
        run_log=run,
    )
    optimizer = optim.get_optim(
        model.params,
        cf.optim,
        cf.lr,
        batch_scale=cf.batch_scale,
        grad_clip=cf.grad_clip,
        weight_decay=cf.weight_decay,
    )

    params = cf  # {"learning_rate": 0.001, "optimizer": "Adam"}
    run["parameters"] = params


    # writer = SummaryWriter(f"{cf.logdir}/tensorboard")
    # # get some random training images
    # images, labels = train_loader[0]
    # images_sq = torch.reshape(images, (6, 1, 28, 28))
    #
    # # create grid of images
    # img_grid = torchvision.utils.make_grid(images_sq)
    #
    # # show images
    # matplotlib_imshow(img_grid, one_channel=True)
    #
    # # write to tensorboard
    # writer.add_image('six_fashion_mnist_images_b', img_grid)
    #
    # writer.close()

    with torch.no_grad():
        metrics = {"acc": []}
        for epoch in range(1, cf.n_epochs + 1):

            print(f"\nTrain @ epoch {epoch} ({len(train_loader)} batches)")
            sleep(0.1)
            for batch_id, (img_batch, label_batch) in enumerate(tqdm(train_loader, disable=False)):
                model.train_batch_supervised(
                    img_batch, label_batch, cf.n_train_iters, fixed_preds=cf.fixed_preds_train
                )
                optimizer.step(
                    curr_epoch=epoch,
                    curr_batch=batch_id,
                    n_batches=len(train_loader),
                    batch_size=img_batch.size(0),
                )

            if epoch % cf.test_every == 0:
                print(f"\nTest @ epoch {epoch}")
                sleep(0.1)
                acc = 0
                for _, (img_batch, label_batch) in enumerate(tqdm(test_loader, disable=False)):
                    label_preds = model.test_batch_supervised(img_batch)
                    acc += datasets.accuracy(label_preds, label_batch)
                metrics["acc"].append(acc / len(test_loader))
                print("\nAccuracy: {:.4f}".format(acc / len(test_loader)))

            utils.save_json(metrics, cf.logdir + "metrics.json")

            run["train/acc"].log(acc / len(test_loader))

    run["eval/f1_score"] = 0.66  # An example only

    run.stop()


if __name__ == "__main__":
    cf = utils.AttrDict()
    cf.seeds = [0]

    for seed in cf.seeds:

        # experiment params
        cf.seed = seed
        cf.n_epochs = 30
        cf.test_every = 1
        cf.log_every = 100
        cf.logdir = f"data/supervised/{seed}/"

        # dataset params
        cf.train_size = None
        cf.test_size = None
        cf.label_scale = None
        cf.normalize = False

        # optim params
        cf.optim = "Adam"
        cf.lr = 5e-3
        cf.batch_size = 6400
        cf.batch_scale = False
        cf.grad_clip = 50
        cf.weight_decay = None

        # inference params
        cf.mu_dt = 0.01
        cf.n_train_iters = 200
        cf.fixed_preds_train = False

        # model params
        cf.use_bias = True
        cf.kaiming_init = False
        cf.nodes = [784, 300, 100, 10]
        cf.act_fn = utils.ReLU()
        cf.use_precis = False
        # cf.precis = [1.0, 625.0, 100.0, 500.0]
        cf.precis = [1.0, 1.0, 1.0, 1.0]

        main(cf)
