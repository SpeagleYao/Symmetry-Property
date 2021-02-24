import torch
import torch.nn.functional as F

def fgsm(img, label, model, criterion=F.cross_entropy, eps=0.007, target_setting=False):
    adv = img.clone().detach()
    adv.requires_grad = True
    if adv.grad is not None:
        adv.grad.data.zero_()

    out = model(adv)
    try:
        loss = criterion(out, label)
    except:
        loss = criterion(out[-1], label)
    loss.backward()

    noise = adv.grad

    if target_setting:
        adv.data = adv.data - eps * noise.sign()
    else:
        adv.data = adv.data + eps * noise.sign()
    adv.data.clamp_(0.0, 1.0)

    l2 = torch.norm((adv - img).reshape(img.shape[0], -1), dim=1).mean()
    return adv.detach(), l2


def pgd(img, label, model, criterion=F.cross_entropy, eps=0.03, iters=10,step=0.007, target_setting=False):
    adv = img.clone().detach()
    adv.requires_grad = True
    if adv.grad is not None:
        adv.grad.data.zero_()

    for j in range(iters):
        out_adv = model(adv)
        try:
            loss = criterion(out_adv, label)
        except:
            loss = criterion(out_adv[-1], label)
        loss.backward()

        noise = adv.grad
        if target_setting:
            adv.data = adv.data - step * noise.sign()
        else:
            adv.data = adv.data + step * noise.sign()

        adv.data = torch.where(adv.data > img.data + eps, img.data + eps, adv.data)
        adv.data = torch.where(adv.data < img.data - eps, img.data - eps, adv.data)
        adv.data.clamp_(0.0, 1.0)
        adv.grad.data.zero_()
    l2 = torch.norm((adv - img).reshape(img.shape[0], -1), dim=1).mean()

    return adv.detach(),l2


def mim(img, label, model, criterion=F.cross_entropy, eps=0.03, iters=10, target_setting=False):
    adv = img.clone().detach()
    adv.requires_grad = True
    if adv.grad is not None:
        adv.grad.data.zero_()

    iterations = iters
    step = eps / iterations
    noise = 0

    for j in range(iterations):
        out_adv = model(adv)
        try:
            loss = criterion(out_adv, label)
        except:
            loss = criterion(out_adv[-1], label)
        loss.backward()

        adv_mean = torch.mean(torch.abs(adv.grad), dim=1, keepdim=True)
        adv_mean = torch.mean(torch.abs(adv_mean), dim=2, keepdim=True)
        adv_mean = torch.mean(torch.abs(adv_mean), dim=3, keepdim=True)
        adv.grad = adv.grad / adv_mean
        noise = noise + adv.grad

        if target_setting:
            adv.data = adv.data - step * noise.sign()
        else:
            adv.data = adv.data + step * noise.sign()

        adv.data.clamp_(0.0, 1.0)
        adv.grad.data.zero_()
    l2 = torch.norm((adv - img).reshape(img.shape[0], -1), dim=1).mean()

    return adv.detach(),l2



def cw(inputs, targets, model, iters=1000, kappa=0, c=3, lr=1, target_setting=False):

    def f(x):
        outputs = model(x)[-1]
        one_hot_labels = torch.eye(len(outputs[0]))[targets].cuda()
        i, _ = torch.max((1 - one_hot_labels) * outputs, dim=1)
        j = torch.masked_select(outputs, one_hot_labels.byte())
        if target_setting:
            return torch.clamp(i - j, min=-kappa)
        else:
            return torch.clamp(j - i, min=-kappa)

    w = torch.zeros_like(inputs, requires_grad=True).cuda()
    optimizer = optim.Adam([w], lr=lr)
    prev = 1e10
    for step in range(iters):
        a = 1 / 2 * (nn.Tanh()(w) + 1)
        loss1 = nn.MSELoss(reduction='sum')(a, inputs)
        loss2 = torch.sum(c * f(a))
        cost = loss1 + loss2
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
       # Early Stop when loss does not converge.
        if step % (iters // 10) == 0:
            if cost > prev:
                print('Attack Stopped due to CONVERGENCE....')
                return a, torch.norm((a - inputs).reshape(inputs.shape[0], -1), dim=1).mean()
            prev = cost

    adv = 1 / 2 * (nn.Tanh()(w) + 1)
    l2 = torch.norm((adv- inputs).reshape(inputs.shape[0], -1), dim=1).mean()

    return adv, l2