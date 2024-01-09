import os
import torch
from torch.autograd import Variable
import numpy as np
import torchvision.transforms as transforms
from torchvision import utils as v_utils
import torchvision
from utils import loss_vc


class AVIH(object):
    def __init__(self, opt):
        self.opt = opt
        self.alpha = 0.01

    def initialization(self, data):
        if self.opt.init == 'random':
            adv = torch.randn_like(data)
        elif self.opt.init == 'original':
            adv = data
        else:
            print('Initialization method not recognized!')
            raise Exception
        return adv

    def get_feature(self, model_r, data):
        if self.opt.src_model == 'AdaFace':
            fea, _ = model_r.forward(data)
        else:
            fea = model_r.forward(data*255)
        return fea

    def attack(self, data, model, model_r):
        opt = self.opt
        # initialization
        adv = self.initialization(data)
        adv = adv.detach().clone().requires_grad_(True)
        input_ga = model.generate(data)
        input_ga = torch.clamp(input_ga, min=0, max=1)
        input_ga = Variable(input_ga)

        sum_grad = torch.zeros_like(input_ga)
        Loss_c = loss_vc().cuda()
        alpha = self.alpha
        tmp_losses = []  # record loss
        num_lo = 0
        cos = 0
        feature = self.get_feature(model_r, data)
        for i in range(opt.num_iter):
            adv = adv.detach().clone().cuda().requires_grad_(True)
            adv_ga = model.generate(adv)
            adv_ga = torch.clamp(adv_ga, min=0, max=1)
            adv_feature = self.get_feature(model_r, adv)
            model.zero_grad_o()
            model_r.zero_grad()

            loss_f_i = torch.mean((adv_feature - feature) ** 2)
            loss_i = torch.mean((adv_ga - data) ** 2)
            # loss_c = -torch.mean((adv - data) ** 2)
            loss_c = Loss_c(adv)
            loss = loss_f_i + 0.2 * loss_i + opt.c * loss_c  # 0.2
            loss.backward(retain_graph=True)
            grad = adv.grad.data.clone()

            grad = grad / grad.abs().mean(dim=[1, 2, 3], keepdim=True)
            sum_grad = 0.3 * sum_grad + grad
            adv.grad.data.zero_()
            adv = adv.data.clone()
            adv = adv - sum_grad * alpha  # torch.sign(sum_grad) * alpha

            adv = torch.clamp(adv, min=0, max=1)
            tmp_losses.append(loss.data.unsqueeze(0).cpu().detach().numpy())
            if i > 1 and tmp_losses[i] > tmp_losses[i - 1]:
                num_lo = num_lo + 1
                if num_lo == 16:
                    alpha = alpha * 0.85
                    num_lo = 0
            if i % 60 == 0 or i == opt.num_iter - 1 or i == 1000 - 1:
                adv_feature = self.get_feature(model_r, adv)
                cos = torch.cosine_similarity(adv_feature, feature).cpu().detach().numpy()
                print('loss:', loss.cpu().detach().numpy(), '--loss_i', loss_i.cpu().detach().numpy(), '--loss_c',
                      loss_c.cpu().detach().numpy(),
                      '--loss_f', loss_f_i.cpu().detach().numpy(), '--cos', cos)
        return adv

    def attack_class(self, data, label, net, model):
        opt = self.opt
        trans = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

        adv = self.initialization(data)
        adv = adv.detach().clone().requires_grad_(True)

        o_mask = torch.nn.functional.one_hot(label, 10).cuda()
        input_ga = model.generate(data)
        input_ga = torch.clamp(input_ga, min=0, max=1)
        input_ga = Variable(input_ga)

        sum_grad = torch.zeros_like(input_ga)
        alpha = 0.02
        Loss_c = loss_vc().cuda()
        tmp_losses = []
        num_lo = 0
        for i in range(opt.num_iter):

            adv = adv.detach().clone().cuda().requires_grad_(True)  # adv
            adv_ga = model.generate(adv)
            adv_ga = torch.clamp(adv_ga, min=0, max=1)
            # adv_feature = model_r.forward(adv * 255)
            out_adv = net(trans(adv))
            model.zero_grad_o()

            id = out_adv * o_mask
            id = torch.mean(id)
            loss_si = Loss_c(adv)
            loss_i = torch.mean((adv_ga - data) ** 2)

            loss = 2 * loss_i + 0.005 * loss_si - 0.001 * id
            loss.backward(retain_graph=True)
            grad = adv.grad.data.clone()
            grad = grad / grad.abs().mean(dim=[1, 2, 3], keepdim=True)
            sum_grad = 0.3 * sum_grad + grad
            adv.grad.data.zero_()
            adv = adv.data.clone()
            adv = adv - sum_grad * alpha
            adv = torch.clamp(adv, min=0, max=1)
            tmp_losses.append(loss.data.unsqueeze(0).cpu().detach().numpy())
            if i > 1 and tmp_losses[i] > tmp_losses[i - 1]:
                num_lo = num_lo + 1
                if num_lo == 8:
                    alpha = alpha * 0.80
                    num_lo = 0
            if i % 20 == 0 or i == opt.num_iter - 1:
                print('loss:', loss.cpu().detach().numpy(), '--loss_i', loss_i.cpu().detach().numpy(), '--loss_si',
                      loss_si.cpu().detach().numpy(), '--loss_c', id.cpu().detach().numpy())
        return adv

    def encrypted(self, data, label, model_g, net):
        if self.opt.task == 'face_recognition':
            adv = self.attack(data, model_g, net)
        elif self.opt.task == 'classification':
            adv = self.attack_class(data, label, net, model_g)
        else:
            print('Unable to identify the target task!')
            raise Exception
        return adv
