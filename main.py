import os
import torch
from options.test_options import TestOptions
from get_model import getmodel
from models.pix2pix_model import Pix2PixModel
import numpy as np
import torchvision.transforms as transforms
from torchvision import utils as v_utils
import torchvision
from avih import AVIH


def main():
    opt = TestOptions().parse()  # get test options
    model_r, img_shape = getmodel(opt.src_model)
    data_loader = get_dataloader(opt)

    model_g = Pix2PixModel(opt)
    model_g.setup(opt)
    num = len(data_loader)
    avh = AVIH(opt)
    print('The total number of data is ', num)

    for i, (data, name_n) in enumerate(data_loader):
        print('---', i, '---')
        data = data.cuda()
        out_o = model_g.generate(data)
        out_o = torch.clamp(out_o, min=0, max=1)
        # Encrypted images
        att = avh.encrypted(data, name_n, model_g, model_r)
        # Decrypted images
        out = model_g.generate(att)
        out = torch.clamp(out, min=0, max=1)
        # Save encrypted images and decrypted images
        all_picture = torch.cat([data, out_o, att, out], dim=0)
        name = name_n.cpu().detach().numpy()
        if not os.path.exists(opt.output):
            os.makedirs(opt.output)
        v_utils.save_image(all_picture, opt.output + str(i) + '.png', normalize=True, padding=0, nrow=opt.batch_size)


def get_dataloader(opt):
    transform = transforms.Compose([
        # transforms.Resize(img_shape),
        transforms.ToTensor(),
    ])
    if opt.task == 'face_recognition':
        dataset = torchvision.datasets.ImageFolder(opt.root, transform=transform)
    elif opt.task == 'classification':
        dataset = torchvision.datasets.CIFAR10(opt.root, train=False, download=False,
                                               transform=transform)
    else:
        print('Unable to identify the target task!')
        raise Exception
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)
    return data_loader


if __name__ == '__main__':
    main()
