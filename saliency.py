## SALIENCY MAP-MAKING

import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np

# Saliency Imports
from captum.attr import Saliency
from captum.attr import visualization as viz
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, Subset, DataLoader, ConcatDataset

class SalGenArgs:
    algorithm = "foster"
    dataset = "cifar100"
    args = None
    desired_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    class_per_task = 10
    num_class = 100
    distill = False

device = "cuda" if torch.cuda.is_available() else "cpu"


def generate_predictions(algorithm, model, ses, images, **kwargs):
    # test the model.
    #pred = model.solve(images)

    # run the model on the real data.
    real_scores = model.forward(images)

    #real_loss = model.criterion(real_scores, y)
    #scoreSubset = real_scores[:, :(ses+1)*2]
    _, pred = real_scores[:, 0: SalGenArgs.class_per_task * (1 + ses)].max(1)
    #real_prec = (y == real_predicted).sum().item() / batch_size
    predicted = pred.squeeze()

    return predicted


def get_indices(dataset, class_name):
    indices = []
    for i in range(len(dataset.targets)):
        for j in class_name:
            if dataset.targets[i] == j:
                indices.append(i)
    return indices


'''def load_model(algorithm, dataset, ses, **kwargs):
    model = None
    model_path = f'checkpoints/split-mnist-generative-replay-r0.3_ses_{ses}'
    print(model_path)
    #cnn = CNN(image_size=32, image_channel_size=1, classes=(ses+1)*2,
    #    depth=5, channel_size=1024, reducing_layers=3)
    cnn = CNN(image_size=32, image_channel_size=1, classes=10,
              depth=5, channel_size=1024, reducing_layers=3)
    wgan = WGAN(z_size=100, image_size=32, image_channel_size=1,
        c_channel_size=64, g_channel_size=64)

    scholar = Scholar('', generator=wgan, solver=cnn, dataset_config=None)

    model_data = torch.load(model_path, map_location=device, weights_only=False)

    #print(model_data['state'].keys())
    #print(model_data['state_dict'].keys())

    #print(scholar.state_dict().keys())
    scholar.load_state_dict(model_data['state'])
    #for _ in range(40): model_data['state'].popitem(last=False)
    #model.load_state_dict(model_data['state'])
    #model.eval()

    return scholar.solver
'''


def load_saliency_data(dataset, desired_classes, imgs_per_class):
    transform = transforms.Compose(
        [transforms.ToTensor()])

    if not os.path.isdir(f"SaliencyMaps/{SalGenArgs.algorithm}/" + SalGenArgs.dataset):
        os.makedirs(f"SaliencyMaps/{SalGenArgs.algorithm}/" + SalGenArgs.dataset)

    saliency_set = torch.utils.data.Dataset()
    mean = None
    std = None
    if dataset == "mnist":
        saliency_set = datasets.MNIST(root=f"../Datasets/{dataset}/", train=False,
                                     download=True,
                                     transform=transforms.Compose([transforms.ToTensor(),
                                                                   transforms.ToPILImage(),
                                                                   transforms.Pad(2),
                                                                   transforms.ToTensor()]))


    elif dataset == "svhn":
        saliency_set = datasets.SVHN(root=f"../Datasets/{dataset}/", split='test',
                                    download=True,
                                    transform=transforms.Compose(
                                        [transforms.ToTensor(),
                                         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]))
        saliency_set.classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        saliency_set.targets = saliency_set.labels
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])

    elif dataset == "cifar10":
        saliency_set = datasets.CIFAR10(root=f"../Datasets/{dataset}/", train=False,
                                       download=True,
                                       transform=transforms.Compose(
                                           [transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
        mean = torch.tensor([0.4914, 0.4822, 0.4465])
        std = torch.tensor([0.2023, 0.1994, 0.2010])

    elif dataset == "cifar100":
        saliency_set = datasets.CIFAR100(root=f"../Datasets/{dataset}/", train=False,
                                        download=True,
                                        transform=transforms.Compose(
                                            [transforms.ToTensor(),
                                             transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))]))
        mean = torch.tensor([0.5071, 0.4867, 0.4408])
        std = torch.tensor([0.2675, 0.2565, 0.2761])

    idx = get_indices(saliency_set, desired_classes)

    subset = Subset(saliency_set, idx)

    # Create a DataLoader for the subset
    saliencyLoader = DataLoader(subset, batch_size=256)

    images, labels = next(iter(saliencyLoader))

    # Order saliency images and labels by class
    sal_idx = []
    sal_labels = []
    for i in range(len(desired_classes)):
        num = 0
        while len(sal_idx) < imgs_per_class * (i + 1):
            if labels[num] == desired_classes[i]:
                sal_idx.append(num)
                sal_labels.append(desired_classes[i])
            num += 4
            #num += 16
            #num += 6
    sal_imgs = images[sal_idx]

    return sal_imgs, torch.tensor(sal_labels), saliency_set.classes, mean, std



def create_saliency_map(model, ses, dataset, desired_classes, imgs_per_class, inline=False, imgs=None, labels=None, preds=None, algorithm=None):
    #if not validate:
    sal_imgs, sal_labels, classes, MEAN, STD = load_saliency_data(dataset, desired_classes, imgs_per_class)
    if imgs is not None: sal_imgs, sal_labels = imgs, labels
    if algorithm is None: algorithm = SalGenArgs.algorithm

    #else:
    #    sal_imgs, sal_labels, classes, MEAN, STD = load_validation_data()

    # Reshape MNIST data for RPSnet
    #if SalGenArgs.algorithm == "RPSnet" and SalGenArgs.dataset == "mnist":
    #    sal_imgs = sal_imgs.detach().numpy().reshape(-1, 784)
    #    sal_imgs = torch.from_numpy(sal_imgs)

    sal_imgs, sal_labels = sal_imgs.to(device), sal_labels.to(device)

    # Add path argument for RPSnet
    #if SalGenArgs.algorithm == "RPSnet":
    #    infer_path = generate_path(ses, SalGenArgs.dataset, SalGenArgs.args)
    #    predicted = generate_predictions(SalGenArgs.algorithm, model, ses, sal_imgs, infer_path=infer_path)
    #else:
    if inline: predicted = preds.squeeze()#predicted = generate_predictions(SalGenArgs.algorithm, model, ses, sal_imgs)
    else: predicted = generate_predictions(algorithm, model, ses, sal_imgs)

    #saliency = Saliency(model)
    saliency = Saliency(lambda x: model(x)["logits"])

    compare_grads = {}
    nrows, ncols = (2, 10) if SalGenArgs.dataset == "cifar100" else (2, 2 * imgs_per_class)
    fig, ax = plt.subplots(nrows, ncols, figsize=(15, 5))
    for ind in range(len(desired_classes) * imgs_per_class):
        compare_grads[ind] = {"grad": None, "original": None, "pred": None}
        image = sal_imgs[ind].unsqueeze(0)
        image.requires_grad = True

        # Add additional arguments for RPSnet
        #if algorithm == "RPSnet":
        #    grads = saliency.attribute(image, target=predicted[ind], abs=False,
        #                               additional_forward_args=(infer_path, -1))
        #else:
        #grads = saliency.attribute(image, target=predicted[ind], abs=False)
        grads = saliency.attribute(image, target=sal_labels[ind], abs=False)

        if SalGenArgs.dataset == "mnist":
            # Reshape MNIST data from RPSnet
            if algorithm == "RPSnet":
                #grads = grads.reshape(28, 28)
                grads = grads.reshape(32, 32)
            else:
                grads = grads.squeeze().cpu().detach()
            squeeze_grads = torch.unsqueeze(grads, 0)
            # Save gradient for comparison
            compare_grads[ind]["grad"] = grads
            grads = np.transpose(squeeze_grads.cpu().numpy(), (1, 2, 0))
        else:
            # Save gradient for comparison
            compare_grads[ind]["grad"] = grads
            grads = np.transpose(grads.squeeze().cpu().detach().numpy(), (1, 2, 0))

        truthStr = 'Truth: ' + str(classes[sal_labels[ind]])
        predStr = 'Pred: ' + str(classes[predicted[ind]])
        print(truthStr + '\n' + predStr)

        # Reshape MNIST data from RPSnet
        if algorithm == "RPSnet" and SalGenArgs.dataset == "mnist":
            #original_image = sal_imgs[ind].cpu().reshape(28, 28).unsqueeze(0)
            original_image = sal_imgs[ind].cpu().reshape(32, 32).unsqueeze(0)
        else:
            original_image = sal_imgs[ind].cpu()

        # Denormalization for RGB datasets
        if SalGenArgs.dataset != "mnist":
            original_image = original_image * STD[:, None, None] + MEAN[:, None, None]

        # Save image for comparison
        compare_grads[ind]["original"] = original_image
        original_image = np.transpose(original_image.detach().numpy(), (1, 2, 0))

        methods = ["original_image", "blended_heat_map"]
        #methods = ["original_image", "heat_map"]
        signs = ["all", "absolute_value"]
        titles = ["Original Image", "Saliency Map"]
        colorbars = [False, True]

        # Check if image was misclassified
        if predicted[ind] != sal_labels[ind]:
            compare_grads[ind]["pred"] = False
            cmap = "Reds"
        else:
            compare_grads[ind]["pred"] = True
            cmap = "Blues"

        # Select row and column for saliency image
        if SalGenArgs.dataset == "cifar100" and ind > 4:
            row, col = (1, ind - 5)
        elif SalGenArgs.dataset != "cifar100" and ind > imgs_per_class - 1:
            row, col = (1, ind - imgs_per_class)
        else:
            row, col = (0, ind)

        # Generate original images and saliency images
        for i in range(2):
            # print(f"Ind: {ind}\nRow: {row}\nCol: {col}\n")
            plt_fig_axis = (fig, ax[row][(2 * col) + i])
            _ = viz.visualize_image_attr(grads, original_image,
                                         method=methods[i],
                                         sign=signs[i],
                                         fig_size=(4, 4),
                                         plt_fig_axis=plt_fig_axis,
                                         cmap=cmap,
                                         show_colorbar=colorbars[i],
                                         title=titles[i],
                                         use_pyplot=False)
            if i == 0:
                if SalGenArgs.dataset == "mnist":
                    ax[row][2 * col + i].images[0].set_cmap('gray')
                ax[row][2 * col + i].set_xlabel(truthStr)
            else:
                ax[row][2 * col + i].images[-1].colorbar.set_label(predStr)

    fig.tight_layout()
    if algorithm == "DGR" and SalGenArgs.distill:
        fig_save_path = f"SaliencyMaps/{algorithm}/{SalGenArgs.dataset}/distill"
    else:
        fig_save_path = f"SaliencyMaps/{algorithm}/{SalGenArgs.dataset}"
    fig.savefig(f"{fig_save_path}/Sess{ses}SalMap.png")
    plt.close()
    torch.save(compare_grads, f"{fig_save_path}/compare_dict_sess{ses}.pt")
    # fig.show()


def create_saliency_map_inline(model, ses, dataset, desired_classes, imgs_per_class):
    #if not validate:
    sal_imgs, sal_labels, classes, MEAN, STD = load_saliency_data(dataset, desired_classes, imgs_per_class)
    #else:
    #    sal_imgs, sal_labels, classes, MEAN, STD = load_validation_data()

    # Reshape MNIST data for RPSnet
    if SalGenArgs.algorithm == "RPSnet" and SalGenArgs.dataset == "mnist":
        sal_imgs = sal_imgs.detach().numpy().reshape(-1, 784)
        sal_imgs = torch.from_numpy(sal_imgs)

    sal_imgs, sal_labels = sal_imgs.to(device), sal_labels.to(device)

    # Add path argument for RPSnet
    #if SalGenArgs.algorithm == "RPSnet":
    #    infer_path = generate_path(ses, SalGenArgs.dataset, SalGenArgs.args)
    #    predicted = generate_predictions(SalGenArgs.algorithm, model, ses, sal_imgs, infer_path=infer_path)
    #else:
    predicted = generate_predictions(SalGenArgs.algorithm, model, ses, sal_imgs)

    saliency = Saliency(model)

    compare_grads = {}
    nrows, ncols = (2, 10) if SalGenArgs.dataset == "cifar100" else (2, 2 * imgs_per_class)
    fig, ax = plt.subplots(nrows, ncols, figsize=(15, 5))
    for ind in range(len(desired_classes) * imgs_per_class):
        compare_grads[ind] = {"grad": None, "original": None, "pred": None}
        image = sal_imgs[ind].unsqueeze(0)
        image.requires_grad = True

        # Add additional arguments for RPSnet
        #if SalGenArgs.algorithm == "RPSnet":
        #    grads = saliency.attribute(image, target=predicted[ind], abs=False,
        #                               additional_forward_args=(infer_path, -1))
        #else:
        #grads = saliency.attribute(image, target=predicted[ind], abs=False)
        grads = saliency.attribute(image, target=sal_labels[ind], abs=False)

        if SalGenArgs.dataset == "mnist":
            # Reshape MNIST data from RPSnet
            if SalGenArgs.algorithm == "RPSnet":
                grads = grads.reshape(28, 28)
            else:
                grads = grads.squeeze().cpu().detach()
            squeeze_grads = torch.unsqueeze(grads, 0)
            # Save gradient for comparison
            compare_grads[ind]["grad"] = grads
            grads = np.transpose(squeeze_grads.numpy(), (1, 2, 0))
        else:
            # Save gradient for comparison
            compare_grads[ind]["grad"] = grads
            grads = np.transpose(grads.squeeze().cpu().detach().numpy(), (1, 2, 0))

        truthStr = 'Truth: ' + str(classes[sal_labels[ind]])
        predStr = 'Pred: ' + str(classes[predicted[ind]])
        print(truthStr + '\n' + predStr)

        # Reshape MNIST data from RPSnet
        if SalGenArgs.algorithm == "RPSnet" and SalGenArgs.dataset == "mnist":
            original_image = sal_imgs[ind].cpu().reshape(28, 28).unsqueeze(0)
        else:
            original_image = sal_imgs[ind].cpu()

        # Denormalization for RGB datasets
        if SalGenArgs.dataset != "mnist":
            original_image = original_image * STD[:, None, None] + MEAN[:, None, None]

        # Save image for comparison
        compare_grads[ind]["original"] = original_image
        original_image = np.transpose(original_image.detach().numpy(), (1, 2, 0))

        methods = ["original_image", "blended_heat_map"]
        signs = ["all", "absolute_value"]
        titles = ["Original Image", "Saliency Map"]
        colorbars = [False, True]

        # Check if image was misclassified
        if predicted[ind] != sal_labels[ind]:
            compare_grads[ind]["pred"] = False
            cmap = "Reds"
        else:
            compare_grads[ind]["pred"] = True
            cmap = "Blues"

        # Select row and column for saliency image
        if SalGenArgs.dataset == "cifar100" and ind > 4:
            row, col = (1, ind - 5)
        elif SalGenArgs.dataset != "cifar100" and ind > imgs_per_class - 1:
            row, col = (1, ind - imgs_per_class)
        else:
            row, col = (0, ind)

        # Generate original images and saliency images
        for i in range(2):
            # print(f"Ind: {ind}\nRow: {row}\nCol: {col}\n")
            plt_fig_axis = (fig, ax[row][(2 * col) + i])
            _ = viz.visualize_image_attr(grads, original_image,
                                         method=methods[i],
                                         sign=signs[i],
                                         fig_size=(4, 4),
                                         plt_fig_axis=plt_fig_axis,
                                         cmap=cmap,
                                         show_colorbar=colorbars[i],
                                         title=titles[i],
                                         use_pyplot=False)
            if i == 0:
                if SalGenArgs.dataset == "mnist":
                    ax[row][2 * col + i].images[0].set_cmap('gray')
                ax[row][2 * col + i].set_xlabel(truthStr)
            else:
                ax[row][2 * col + i].images[-1].colorbar.set_label(predStr)

    fig.tight_layout()
    if SalGenArgs.algorithm == "DGR" and SalGenArgs.distill:
        fig_save_path = f"SaliencyMaps/{SalGenArgs.algorithm}/{SalGenArgs.dataset}/distill"
    else:
        fig_save_path = f"SaliencyMaps/{SalGenArgs.algorithm}/{SalGenArgs.dataset}"
    fig.savefig(f"{fig_save_path}/Sess{ses}SalMap.png")
    plt.close()
    torch.save(compare_grads, f"{fig_save_path}/compare_dict_sess{ses}.pt")
    # fig.show()