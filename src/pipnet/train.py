###########################################################################
###                               PIPNet                                ###
###                         Training functions                          ###
###                        Author: Manuel Cordova                       ###
###                       Last edited: 2021-10-12                       ###
###########################################################################

import os
import torch
import torch.nn as nn
import numpy as np

def evaluate(
    data_generator,
    net,
    loss,
    epoch,
    batches_per_eval=10,
    avg_models=False,
    out_dir=None,
    device="cpu",
    monitor_end="\n",
):

    net.eval()

    val_losses = []
    val_components = []
    # Dummy loss components if not returned
    components = [0., 0.]

    # Evaluation loop
    for val_batch, (X, _, y) in enumerate(data_generator):

        X = X.to(device)
        y = y.to(device)

        # Forward pass
        with torch.no_grad():
            y_pred, y_std, ys_pred = net(X)

        if net.return_all_layers:
            if net.ndim == 1:
                y = y.repeat((1, y_pred.shape[1], 1))
            elif net.ndim == 2:
                y = y.repeat((1, y_pred.shape[1], 1, 1))

        # Compute loss
        if not net.is_ensemble or avg_models:
            if loss.return_components:
                l, components = loss(y_pred, y)
            else:
                l = loss(y_pred, y)
        else:
            ys = torch.cat([torch.unsqueeze(y.clone(), 0) for _ in range(ys_pred.shape[0])])
            if loss.return_components:
                l, components = loss(ys_pred, ys)
            else:
                l = loss(ys_pred, ys)

        # Update monitoring lists
        val_losses.append(float(l.detach()))
        val_components.append(components)

        pp = "    Validation batch {: 4d}: ".format(val_batch + 1)
        pp += "loss = {: 1.4e}, ".format(val_losses[-1])
        pp += "mean loss = {: 1.4e}...".format(np.mean(val_losses))
        print(pp, end=monitor_end)

        if (val_batch + 1) >= batches_per_eval:
            break

    net.train()

    if out_dir is not None:
        torch.save(net.state_dict(), out_dir + f"epoch_{epoch}_network")
        np.save(out_dir + f"epoch_{epoch}_in.npy", X.cpu().numpy())
        np.save(out_dir + f"epoch_{epoch}_trg.npy", y.cpu().numpy())
        np.save(out_dir + f"epoch_{epoch}_pred.npy", y_pred.detach().cpu().numpy())
        if net.is_ensemble:
            np.save(out_dir + f"epoch_{epoch}_std.npy", y_std.detach().cpu().numpy())

    return val_losses, val_components

def train(
    dataset,
    net,
    opt,
    loss,
    sch,
    batch_size=64,
    batches_per_epoch=100,
    batches_per_eval=10,
    n_epochs=10,
    avg_models=False,
    change_loss={},
    out_dir=None,
    num_workers=1,
    device="cpu",
    monitor_end="\n",
):
    """
    Train the model

    Inputs: - dataset           Training (and test) data generator
            - net               Network
            - opt               Optimizer
            - loss              Loss function
            - sch               Learning rate scheduler
            - train_pars        Training parameters
    """

    # Create output directory if needed
    if out_dir is not None and not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # Initialize data generator
    data_generator = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    all_losses = []
    all_val_losses = []
    all_lrs = []
    all_components = []
    all_val_components = []

    losses = []
    lrs = []
    components = []
    # Dummy loss components if not returned
    these_components = [0., 0.]

    epoch = 1

    print("Starting training...")

    # Training loop
    for batch, (X, _, y) in enumerate(data_generator):

        X = X.to(device)
        y = y.to(device)

        # Zero out the parameter gradients
        opt.zero_grad()

        # Forward pass
        y_pred, y_std, ys_pred = net(X)

        if net.return_all_layers:
            if net.ndim == 1:
                y = y.repeat((1, y_pred.shape[1], 1))
            elif net.ndim == 2:
                y = y.repeat((1, y_pred.shape[1], 1, 1))

        if batch == 0:
            assert(y_pred.shape == y.shape)

        # Compute loss
        if not net.is_ensemble or avg_models:
            if loss.return_components:
                l, these_components = loss(y_pred, y)
            else:
                l = loss(y_pred, y)
        else:
            ys = torch.cat([torch.unsqueeze(y.clone(), 0) for _ in range(ys_pred.shape[0])])
            if loss.return_components:
                l, these_components = loss(ys_pred, ys)
            else:
                l = loss(ys_pred, ys)

        # Backward pass
        l.backward()

        # Optimizer step
        opt.step()

        # Update monitoring lists
        losses.append(float(l.detach()))
        lrs.append(opt.param_groups[0]["lr"])
        components.append(these_components)

        pp = "    Training batch {: 4d}: ".format(batch + 1)
        pp += "loss = {: 1.4e}, ".format(losses[-1])
        pp += "mean loss = {: 1.4e}, ".format(np.mean(losses))
        pp += "lr = {: 1.4e}...".format(lrs[-1])
        print(pp, end=monitor_end)

        # End of epoch
        if (batch + 1) % batches_per_epoch == 0:

            print("\n  Checkpoint reached, evaluating the model...")
            val_losses, val_components = evaluate(
                data_generator,
                net,
                loss,
                epoch,
                batches_per_eval=batches_per_eval,
                avg_models=avg_models,
                out_dir=out_dir,
                device=device,
                monitor_end=monitor_end
            )

            all_val_losses.append(val_losses)
            all_losses.append(losses)
            all_lrs.append(lrs)
            all_components.append(components)
            all_val_components.append(val_components)

            losses = []
            lrs = []
            components = []

            if out_dir is not None:
                np.save(out_dir + "all_losses.npy", np.array(all_losses))
                np.save(out_dir + "all_loss_components.npy", np.array(all_components))
                np.save(out_dir + "all_val_losses.npy", np.array(all_val_losses))
                np.save(out_dir + "all_val_loss_components.npy", np.array(all_val_components))
                np.save(out_dir + "all_lrs.npy", np.array(all_lrs))

            # Learning rate scheduler step
            sch.step(np.mean(val_losses))

            print("\n  End of evaluation.")

            # Update loss
            if epoch in change_loss:
                for k in change_loss[epoch]:
                    print(f"\n    Changing loss parameter {k} to {change_loss[epoch][k]}...")
                    loss.update_param(k, change_loss[epoch][k])

            epoch += 1

        if epoch > n_epochs:
            print("End of training.")
            break

    print("All done!")

    return
