###########################################################################
###                               PIPNet                                ###
###                         Training functions                          ###
###                        Author: Manuel Cordova                       ###
###                       Last edited: 2021-10-12                       ###
###########################################################################

import torch
import torch.nn as nn
import numpy as np

def evaluate(data_generator, net, loss, train_pars, i_chk):

    net.eval()

    val_losses = []
    val_components = []
    # Dummy loss components if not returned
    components = [0., 0.]

    # Evaluation loop
    for val_batch, (X, _, y) in enumerate(data_generator):

        X = X.to(train_pars["device"])
        y = y.to(train_pars["device"])

        # Forward pass
        y_pred, y_std, ys_pred = net(X)

        if net.return_all_layers:
            y = y.repeat((1, y_pred.shape[-2], 1))

        # Compute loss
        if not net.is_ensemble or train_pars["avg_models"]:
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
        print(pp, end=train_pars["monitor_end"])

        if (val_batch + 1) >= train_pars["n_eval"]:
            break

    net.train()

    torch.save(net.state_dict(), train_pars["out_dir"] + f"checkpoint_{i_chk}_network")
    np.save(train_pars["out_dir"] + f"checkpoint_{i_chk}_in.npy", X.cpu().numpy())
    np.save(train_pars["out_dir"] + f"checkpoint_{i_chk}_trg.npy", y.cpu().numpy())
    np.save(train_pars["out_dir"] + f"checkpoint_{i_chk}_pred.npy", y_pred.detach().cpu().numpy())
    if net.is_ensemble:
        np.save(train_pars["out_dir"] + f"checkpoint_{i_chk}_std.npy", y_std.detach().cpu().numpy())

    return val_losses, val_components

def train(dataset, net, opt, loss, sch, train_pars):
    """
    Train the model

    Inputs: - dataset           Training (and test) data generator
            - net               Network
            - opt               Optimizer
            - loss              Loss function
            - sch               Learning rate scheduler
            - train_pars        Training parameters
    """

    # Initialize data generator
    data_generator = torch.utils.data.DataLoader(dataset, batch_size=train_pars["batch_size"], shuffle=False, num_workers=train_pars["num_workers"])

    # Set network to training mode
    net.train()

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

    i_chk = 1

    print("Starting training...")

    # Training loop
    for batch, (X, _, y) in enumerate(data_generator):

        X = X.to(train_pars["device"])
        y = y.to(train_pars["device"])

        # Zero out the parameter gradients
        opt.zero_grad()

        # Forward pass
        y_pred, y_std, ys_pred = net(X)

        if net.return_all_layers:
            y = y.repeat((1, y_pred.shape[-2], 1))

        if batch == 0:
            assert(y_pred.shape == y.shape)

        # Compute loss
        if not net.is_ensemble or train_pars["avg_models"]:
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
        print(pp, end=train_pars["monitor_end"])

        # End of epoch
        if (batch + 1) % train_pars["checkpoint"] == 0:

            print("\n  Checkpoint reached, evaluating the model...")
            val_losses, val_components = evaluate(data_generator, net, loss, train_pars, i_chk)

            all_val_losses.append(val_losses)
            all_losses.append(losses)
            all_lrs.append(lrs)
            all_components.append(components)
            all_val_components.append(val_components)

            losses = []
            lrs = []
            components = []

            np.save(train_pars["out_dir"] + "all_losses.npy", np.array(all_losses))
            np.save(train_pars["out_dir"] + "all_loss_components.npy", np.array(all_components))
            np.save(train_pars["out_dir"] + "all_val_losses.npy", np.array(all_val_losses))
            np.save(train_pars["out_dir"] + "all_val_loss_components.npy", np.array(all_val_components))
            np.save(train_pars["out_dir"] + "all_lrs.npy", np.array(all_lrs))

            # Learning rate scheduler step
            sch.step(np.mean(val_losses))

            # Update loss
            if i_chk in train_pars["change_factor"]:
                loss.factor = train_pars["change_factor"][i_chk]

            i_chk += 1

            print("\n  End of evaluation.")

        if i_chk > train_pars["max_checkpoints"]:
            print("End of training.")
            break

    print("All done!")

    return
