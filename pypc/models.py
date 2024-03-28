import torch
import numpy as np

from pypc import utils
from pypc.layers import FCLayer


class PCModel(object):
    def __init__(self, nodes, mu_dt, act_fn, use_bias=False, kaiming_init=False, use_precis=False, precis_factor=None, precis_coverage=None, precis_per_pixel=None, run_log=None, log_node_its=False):
        """
        Define the Predictive Coding PyTorch model. All layers fully connected. All nodes using the specified activation
        function except for the output layer which is linear. Bias terms are optional. Kaiming weight initialisation is
        optional. Precisions are optional.

        :param nodes: List of number of nodes in each layer
        :param mu_dt: Timestep for updating means
        :param act_fn: Activation function
        :param use_bias: Include bias terms?
        :param kaiming_init: Use Kaiming weight initialisation?
        :param use_precis: Use precisions? (If False, precisions are implied to be identity matrices)
        :param precis_factor: List of precision scaling per node layer
        :param precis_coverage: List of precision coverage by node layer or None for full coverage
        :param run_log: A neptune.ai run object or None if logging not required
        :param log_node_its: Log stats after every node update iteration? (WARNING: Slow!)
        """

        self.nodes = nodes
        self.mu_dt = mu_dt

        self.n_nodes = len(nodes)
        self.n_layers = len(nodes) - 1

        self.layers = []
        for l in range(self.n_layers):
            _act_fn = utils.Linear() if (l == self.n_layers - 1) else act_fn

            layer = FCLayer(
                in_size=nodes[l],
                out_size=nodes[l + 1],
                act_fn=_act_fn,
                use_bias=use_bias,
                kaiming_init=kaiming_init,
            )
            self.layers.append(layer)

        self.use_precis = use_precis
        self.precis_coverage = precis_coverage
        # If precisions used, create:
        #   List of diagonal precision matrices with given scale factor
        #   List of diagonal variance matrices (inverse of precision matrices) (easy for diagonal precision)
        #   List of variance determinants (for free energy sum)
        # NOTE: Precisions are fixed, so can be defined here. Not suitable for dynamic (learned) precisions
        if self.use_precis:
            self.precis = [[] for _ in range(self.n_nodes)]
            # self.varis = [[] for _ in range(self.n_nodes)]
            # self.vari_dets = [1.0 for _ in range(self.n_nodes)]
            for n in range(self.n_nodes):
                if (n == self.n_nodes - 1) and precis_per_pixel is not None:
                    temp = precis_per_pixel.flatten()
                else:
                    temp = precis_factor[n] * torch.ones(nodes[n])
                if self.precis_coverage is not None:
                    coverage = max(0.0, min(1.0, self.precis_coverage[n]))  # Clamp to range [0, 1]
                    first_scaled_precision = int(self.nodes[n] * (1.0 - coverage))
                    temp[0:first_scaled_precision] = torch.ones(first_scaled_precision)
                self.precis[n] = utils.set_tensor(torch.diagflat(temp))
                # TODO: Need to rework below to account for precision coverage
                # self.varis[n] = utils.set_tensor(torch.diagflat((1.0/precis_factor[n])*torch.ones(nodes[n])))
                # self.vari_dets[n] = (1.0/precis_factor[n])**nodes[n]
                # self.loggy = np.log(2.0*np.pi*self.vari_dets[n])

        self.run_log = run_log
        self.log_node_its = log_node_its

    def reset(self):
        """
        Initialise predictions (preds), errors (errs), and variational means (mus) to empty lists
        """
        self.preds = [[] for _ in range(self.n_nodes)]
        self.errs = [[] for _ in range(self.n_nodes)]
        self.mus = [[] for _ in range(self.n_nodes)]
        self.free_energy = [0.0 for _ in range(self.n_nodes)]

    def reset_mus(self, batch_size, init_std):
        """
        For each layer, initialise variational means (mus) to be normally distributed with mean=0

        :param batch_size: Number of samples in current batch
        :param init_std: Initial standard deviation of mus
        """
        for l in range(self.n_layers):
            self.mus[l] = utils.set_tensor(
                torch.empty(batch_size, self.layers[l].in_size).normal_(mean=0, std=init_std)
            )

    def set_input(self, inp):
        """
        Set input node mus for batch to input values

        :param inp: Input values, Tensor:(batch_size, nodes[0])
        """
        self.mus[0] = inp.clone()

    def set_target(self, target):
        """
        Set output node mus for batch to target values

        :param target: Target values, Tensor:(batch_size, nodes[-1])
        """
        self.mus[-1] = target.clone()

    def forward(self, val):
        for layer in self.layers:
            val = layer.forward(val)
        return val

    def propagate_mu(self):
        """
        Perform forward pass for batch, update mus for all nodes except inputs and outputs (targets)
        """
        for l in range(1, self.n_layers):
            self.mus[l] = self.layers[l - 1].forward(self.mus[l - 1])

    def train_batch_supervised(self, img_batch, label_batch, n_iters, fixed_preds=False):
        """
        Train the model using the (mini)batch images as inputs and labels as targets

        :param img_batch: Batch of input images, Tensor:(batch_size, nodes[0])
        :param label_batch: Batch of target labels, Tensor:(batch_size, nodes[-1])
        :param n_iters: Number of training iterations
        :param fixed_preds: Fix predictions at initial values?
        """
        self.reset()  # Initialise the prediction, error, and mu data structures
        self.set_input(img_batch)  # Set the model inputs, mus[0], equal to the training *images*
        self.propagate_mu()  # Perform forward pass, update mus for all nodes except inputs and targets
        self.set_target(label_batch)  # Set the model outputs (targets), mus[-1], equal to the training *labels*
        self.train_updates(n_iters, fixed_preds=fixed_preds)  # Iteratively update mus, predictions and errors
        self.update_grads()  # Calculate gradients of weights and biases for all layers

    # def set_precisions_by_per_pixel_variance(self, img_batch):
    #     per_pixel_variance = img_batch.var(dim=0)
    #     # per_pixel_precision = 1.0 / (per_pixel_variance + 1e-1)
    #     per_pixel_precision = 1.0 / per_pixel_variance.clamp(min=0.1)
    #     self.precis[3] = torch.diagflat(per_pixel_precision)

    def train_batch_generative(self, img_batch, label_batch, n_iters, fixed_preds=False, log_batch=False):
        """
        Train the model using the (mini)batch labels as inputs and images as targets
        (Identical to train_batch_supervised() but inputs and targets are swapped)

        :param img_batch: Batch of target images, Tensor:(batch_size, nodes[-1])
        :param label_batch: Batch of input labels, Tensor:(batch_size, nodes[0])
        :param n_iters: Number of training iterations
        :param fixed_preds: Fix predictions at initial values?
        """
        self.reset()  # Initialise the prediction, error, and mu data structures
        self.set_input(label_batch)  # Set the model inputs, mus[0], equal to the training *labels*
        self.propagate_mu()  # Perform forward pass, update mus for all nodes except inputs and targets
        self.set_target(img_batch)  # Set the model outputs (targets), mus[-1], equal to the training *images*
        self.train_updates(n_iters, fixed_preds=fixed_preds, log_batch=log_batch)  # Iteratively update mus, predictions and errors
        self.update_grads()  # Calculate gradients of weights and biases for all layers
        if self.run_log and log_batch:
            self.log_layer_stats(prefix="train")

    def log_layer_stats(self, prefix="layer"):
        for l in range(self.n_layers):
            self.run_log[f"{prefix}/grad_weight_mean_{l}]"].log(torch.mean(self.layers[l].grad["weights"]))
            self.run_log[f"{prefix}/grad_bias_mean_{l}]"].log(torch.mean(self.layers[l].grad["bias"]))
            self.run_log[f"{prefix}/grad_weight_std_{l}]"].log(torch.std(self.layers[l].grad["weights"]))
            self.run_log[f"{prefix}/grad_bias_std_{l}]"].log(torch.std(self.layers[l].grad["bias"]))
            self.run_log[f"{prefix}/weight_mean_{l}]"].log(torch.mean(self.layers[l].weights))
            self.run_log[f"{prefix}/bias_mean_{l}]"].log(torch.mean(self.layers[l].bias))
            self.run_log[f"{prefix}/weight_std_{l}]"].log(torch.std(self.layers[l].weights))
            self.run_log[f"{prefix}/bias_std_{l}]"].log(torch.std(self.layers[l].bias))

    def log_node_stats(self, prefix="node"):
        for n in range(1, self.n_nodes):
            self.run_log[f"{prefix}/err_mean_{n}]"].log(torch.mean(self.errs[n]))
            self.run_log[f"{prefix}/pred_mean_{n}"].log(torch.mean(self.preds[n]))
            self.run_log[f"{prefix}/mu_mean_{n}"].log(torch.mean(self.mus[n]))
            self.run_log[f"{prefix}/err_std_{n}]"].log(torch.std(self.errs[n]))
            self.run_log[f"{prefix}/pred_std_{n}"].log(torch.std(self.preds[n]))
            self.run_log[f"{prefix}/mu_std_{n}"].log(torch.std(self.mus[n]))
            self.run_log[f"{prefix}/free_e_{n}"].log(self.free_energy[n])
        self.run_log[f"{prefix}/free_e"].log(sum(self.free_energy))

    def test_batch_supervised(self, img_batch):
        return self.forward(img_batch)

    def test_batch_generative(self, img_batch, n_iters, init_std=0.05, fixed_preds=False, log_batch=False):
        batch_size = img_batch.size(0)
        self.reset()  # Initialise the prediction, error, and mu data structures
        self.reset_mus(batch_size, init_std)  # Initialise variational means (mus)
        self.set_target(img_batch)  # Set output node mus for batch to target values (test images)
        self.test_updates(n_iters, fixed_preds=fixed_preds, log_batch=log_batch)
        # self.update_grads()  # Calculate gradients of weights and biases for all layers - NOT REQUIRED
        return self.mus[0]

    def train_updates(self, n_iters, fixed_preds=False, log_batch=False):
        """
        Iteratively update mus, predictions and errors

        :param n_iters: Number of training iterations
        :param fixed_preds: Fix predictions at initial values?
        """
        # For batch, initialise predictions and errors for all nodes except inputs
        # Optionally, errors are precision scaled and free energy is calculated
        for n in range(1, self.n_nodes):
            self.preds[n] = self.layers[n - 1].forward(self.mus[n - 1])
            self.errs[n] = self.mus[n] - self.preds[n]
            if self.use_precis:
                self.free_energy[n] = torch.mean(self.errs[n] @ self.precis[n] @ self.errs[n].T).item()
                self.errs[n] = torch.matmul(self.errs[n], self.precis[n])

        # Log values to neptune
        if self.run_log and log_batch and self.log_node_its:
            self.log_node_stats(prefix="train")

        # For each training iteration
        for itr in range(n_iters):
            # For batch, update mus for all nodes except inputs (labels) and outputs (images)
            for l in range(1, self.n_layers):
                delta = self.layers[l].backward(self.errs[l + 1]) - self.errs[l]
                self.mus[l] = self.mus[l] + self.mu_dt * delta

            # For batch, update errors and (optionally) predictions for all nodes except inputs
            # Optionally, errors are precision scaled and free energy is calculated
            for n in range(1, self.n_nodes):
                if not fixed_preds:
                    self.preds[n] = self.layers[n - 1].forward(self.mus[n - 1])
                self.errs[n] = self.mus[n] - self.preds[n]
                if self.use_precis:
                    self.free_energy[n] = torch.mean(self.errs[n] @ self.precis[n] @ self.errs[n].T).item()
                    self.errs[n] = torch.matmul(self.errs[n], self.precis[n])

            # Log values to neptune
            if self.run_log and log_batch and (self.log_node_its or (itr == n_iters-1)):
                self.log_node_stats(prefix="train")

    def test_updates(self, n_iters, fixed_preds=False, log_batch=False):
        """
        Test model

        :param n_iters: Number of training iterations
        :param fixed_preds: Fix predictions at initial values?
        """
        # For batch, initialise predictions and errors for all nodes except inputs
        # Optionally, errors are precision scaled and free energy is calculated
        for n in range(1, self.n_nodes):
            self.preds[n] = self.layers[n - 1].forward(self.mus[n - 1])
            self.errs[n] = self.mus[n] - self.preds[n]
            if self.use_precis:
                self.free_energy[n] = torch.mean(self.errs[n] @ self.precis[n] @ self.errs[n].T).item()
                self.errs[n] = torch.matmul(self.errs[n], self.precis[n])

        # Log values to neptune
        if self.run_log and log_batch and self.log_node_its:
            self.log_node_stats(prefix="test")

        # For each test iteration
        for itr in range(n_iters):
            # For batch, update mus for all nodes except outputs (images)
            # NOTE: Unlike training which also does not update inputs (labels)
            delta = self.layers[0].backward(self.errs[1])
            self.mus[0] = self.mus[0] + self.mu_dt * delta
            for l in range(1, self.n_layers):
                delta = self.layers[l].backward(self.errs[l + 1]) - self.errs[l]
                self.mus[l] = self.mus[l] + self.mu_dt * delta

            # For batch, update errors and (optionally) predictions for all nodes except inputs
            # Optionally, errors are precision scaled and free energy is calculated
            for n in range(1, self.n_nodes):
                if not fixed_preds:
                    self.preds[n] = self.layers[n - 1].forward(self.mus[n - 1])
                self.errs[n] = self.mus[n] - self.preds[n]
                if self.use_precis:
                    self.free_energy[n] = torch.mean(self.errs[n] @ self.precis[n] @ self.errs[n].T).item()
                    self.errs[n] = torch.matmul(self.errs[n], self.precis[n])

            # Log values to neptune
            if self.run_log and log_batch and (self.log_node_its or (itr == n_iters-1)):
                self.log_node_stats(prefix="test")

    def update_grads(self):
        """
        Calculate gradients of weights and biases for all layers

        """
        for l in range(self.n_layers):
            self.layers[l].update_gradient(self.errs[l + 1])

    def get_target_loss(self):
        """
        Calculate loss as the sum of the squares of the target errors
        (Not currently used)

        :return: Loss
        """
        return torch.sum(self.errs[-1] ** 2).item()

    @property
    def params(self):
        """
        Allows controlled and standardised access to model parameters (for passing to an optimizer).
        Currently of limited use but could be expanded.

        :return: Model layers
        """
        return self.layers
