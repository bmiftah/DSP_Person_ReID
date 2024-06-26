from __future__ import print_function, absolute_import
import time

import torch
from torch.autograd import Variable

from .evaluation_metrics import accuracy
from .utils.meters import AverageMeter
from .utils import Bar
from torch.nn import functional as F

def write(path, content):
    with open(path, "a+") as dst_file:
        dst_file.write(content)

class BaseTrainer(object):
    def __init__(self, model, criterion, X, Y, SMLoss_mode=0):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion

    def train(self, epoch, data_loader, optimizer, print_freq=1):
        self.model.train()


        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()
        end = time.time()

        bar = Bar('Processing', max=len(data_loader))
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            inputs, targets = self._parse_data(inputs)
            loss0, loss1, loss2, loss3, loss4, loss5,loss6,loss_layer1,loss_layer2,loss_layer3, prec1 = self._forward(inputs, targets)
#===================================================================================
            loss_layer = (loss_layer1+loss_layer2+loss_layer3+loss6)/4
            loss_ = (loss0+loss1+loss2+loss3+loss4+loss5)/6
            loss = 0.2*loss_layer+0.8*loss_
            # print ('loss:',loss0,loss1, loss2, loss3, loss4, loss5,loss6,loss_layer1,loss_layer2,loss_layer3,)

            losses.update(loss.data, targets.size(0))
            precisions.update(prec1, targets.size(0))
            #one = torch.tensor(1, dtype=torch.float)
            #one = one.to(device)
            # 1, dtype=torch.float
            optimizer.zero_grad()
            #torch.autograd.backward([loss0, loss1, loss2, loss3, loss4, loss5, loss6,loss_layer1,loss_layer2,loss_layer3],
            #                        [torch.ones(1).cuda(), torch.ones(1).cuda(), torch.ones(1).cuda(),torch.ones(1).cuda(),torch.ones(1).cuda(),
            #                        torch.ones(1).cuda(), torch.ones(1).cuda(), torch.ones(1).cuda(),torch.ones(1).cuda(),torch.ones(1).cuda()])

            torch.autograd.backward([loss0, loss1, loss2, loss3, loss4, loss5, loss6,loss_layer1,loss_layer2,loss_layer3],
                                    [torch.tensor(1, dtype=torch.float).cuda(), torch.tensor(1, dtype=torch.float).cuda(), torch.tensor(1, dtype=torch.float).cuda(),torch.tensor(1, dtype=torch.float).cuda(),torch.tensor(1, dtype=torch.float).cuda(),
                                     torch.tensor(1, dtype=torch.float).cuda(), torch.tensor(1, dtype=torch.float).cuda(), torch.tensor(1, dtype=torch.float).cuda(),torch.tensor(1, dtype=torch.float).cuda(),torch.tensor(1, dtype=torch.float).cuda()])
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()
            write('loss.log','epoch:'+str(epoch)+'   loss_avg:'+str(losses.avg)+'    loss_val:'+str(losses.val)+'\r')
            # plot progress
            bar.suffix = 'Epoch: [{N_epoch}][{N_batch}/{N_size}] | Time {N_bt:.3f} {N_bta:.3f} | Data {N_dt:.3f} {N_dta:.3f} | Loss {N_loss:.3f} {N_lossa:.3f} | Prec {N_prec:.2f} {N_preca:.2f}'.format(
                      N_epoch=epoch, N_batch=i + 1, N_size=len(data_loader),
                              N_bt=batch_time.val, N_bta=batch_time.avg,
                              N_dt=data_time.val, N_dta=data_time.avg,
                              N_loss=losses.val, N_lossa=losses.avg,
                              N_prec=precisions.val, N_preca=precisions.avg,
							  )
            bar.next()
        bar.finish()



    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets):
        raise NotImplementedError


class Trainer(BaseTrainer):
    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = Variable(imgs)
        targets = Variable(pids.cuda())
        return inputs, targets

    def _forward(self, inputs, targets):
        #outputs = self.model(*inputs)
        outputs = self.model(inputs)
        index = (targets-751).data.nonzero().squeeze_()
		
        if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
            loss0 = self.criterion(outputs[1][0],targets)
            loss1 = self.criterion(outputs[1][1],targets)
            loss2 = self.criterion(outputs[1][2],targets)
            loss3 = self.criterion(outputs[1][3],targets)
            loss4 = self.criterion(outputs[1][4],targets)
            loss5 = self.criterion(outputs[1][5],targets)
            loss6 = self.criterion(outputs[1][6], targets)

            loss_layer1 = self.criterion(outputs[1][7], targets)
            loss_layer2 = self.criterion(outputs[1][8], targets)
            loss_layer3 = self.criterion(outputs[1][9], targets)
            prec, = accuracy(outputs[1][2].data, targets.data)
            prec = prec
                        
        elif isinstance(self.criterion, OIMLoss):
            loss, outputs = self.criterion(outputs, targets)
            prec, = accuracy(outputs.data, targets.data)
            prec = prec
        elif isinstance(self.criterion, TripletLoss):
            loss, prec = self.criterion(outputs, targets)
        else:
            raise ValueError("Unsupported loss:", self.criterion)
        return loss0, loss1, loss2, loss3, loss4, loss5, loss6, loss_layer1,loss_layer2,loss_layer3, prec
