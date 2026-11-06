import os
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
 # 或 import numpy.core.multiarray
import torch.serialization
from models.networks import *
from misc.metric_tool import ConfuseMatrixMeter
from misc.logger_tool import Logger
from utils import de_norm
import utils
from PIL import Image


def cm2_change_metrics(confusion_matrix):
    hist=confusion_matrix.astype(np.float64)
    eps=np.finfo(np.float32).eps

    TN=hist[0,0]
    FP=hist[0,1]
    FN=hist[1,0]
    TP=hist[1,1]

    Pre=TP/(TP+FP+eps)
    Rec=TP/(TP+FN+eps)
    F1=2*TP/(2*TP+FP+FN+eps)
    IoU=TP/(TP+FP+FN+eps)

    return F1,IoU,Pre,Rec,TP,FP,FN,TN

class CDEvaluator():

    def __init__(self, args, dataloader):

        self.dataloader = dataloader

        self.n_class = args.n_class
        # define G
        self.net_G = define_G(args=args, gpu_ids=args.gpu_ids)
        self.device = torch.device("cuda:%s" % args.gpu_ids[0] if torch.cuda.is_available() and len(args.gpu_ids)>0
                                   else "cpu")
        print(self.device)


        self.running_metric = ConfuseMatrixMeter(n_class=self.n_class)


        logger_path = os.path.join(args.checkpoint_dir, 'log_test.txt')
        self.logger = Logger(logger_path)
        self.logger.write_dict_str(args.__dict__)

        logger_path2 = os.path.join(args.checkpoint_dir, 'log_F1_test.txt')
        self.logger2 = Logger(logger_path2)
        self.logger2.write_dict_str(args.__dict__)

        logger_path3 = os.path.join(args.checkpoint_dir, 'log_PRE_test.txt')
        self.logger3 = Logger(logger_path3)
        self.logger3.write_dict_str(args.__dict__)

        logger_path4 = os.path.join(args.checkpoint_dir, 'log_RECALL_test.txt')
        self.logger4 = Logger(logger_path4)
        self.logger4.write_dict_str(args.__dict__)

        #  training log
        self.epoch_acc = 0
        self.batch_size = args.batch_size
        self.best_val_acc = 0.0
        self.best_epoch_id = 0

        self.steps_per_epoch = len(dataloader)

        self.G_pred = None


        self.pred_vis = None
        self.batch = None
        self.is_training = False
        self.batch_id = 0
        self.epoch_id = 0
        self.checkpoint_dir = args.checkpoint_dir
        self.vis_dir = args.vis_dir

        # check and create model dir
        if os.path.exists(self.checkpoint_dir) is False:
            os.mkdir(self.checkpoint_dir)
        if os.path.exists(self.vis_dir) is False:
            os.mkdir(self.vis_dir)
        if os.path.exists(self.vis_dir + "/predict") is False:
            os.mkdir(self.vis_dir + "/predict")
        if os.path.exists(self.vis_dir + "/label") is False:
            os.mkdir(self.vis_dir + "/label")


    def _load_checkpoint(self, checkpoint_name='best_ckpt.pt'):

        if os.path.exists(os.path.join(self.checkpoint_dir, checkpoint_name)):
            self.logger.write('loading last checkpoint...\n')
            with torch.serialization.safe_globals([np.dtype]):

                checkpoint = torch.load(os.path.join(self.checkpoint_dir, checkpoint_name), map_location=self.device,
                       weights_only=False)
            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])

            self.net_G.to(self.device)

            # update some other states
            self.best_val_acc = checkpoint['best_val_acc']
            self.best_epoch_id = checkpoint['best_epoch_id']

            self.logger.write('Eval Historical_best_acc = %.4f (at epoch %d)\n' %
                  (self.best_val_acc, self.best_epoch_id))
            self.logger.write('\n')

        else:
            raise FileNotFoundError('no such checkpoint %s' % checkpoint_name)


    def _visualize_pred(self):

        pred = torch.where(self.G_pred > 0.5, torch.ones_like(self.G_pred), torch.zeros_like(self.G_pred))
        pred_vis = pred * 255
        return pred_vis

    def _update_metric(self):
        target = self.batch['L'].to(self.device).detach()

        G_pred = torch.where(
            self.G_pred > 0.5,
            torch.ones_like(self.G_pred),
            torch.zeros_like(self.G_pred)
        ).long()

        current_score = self.running_metric.update_cm(
            pr=G_pred.cpu().numpy(),
            gt=target.cpu().numpy()
        )
        return current_score

    def _collect_running_batch_states(self):
        self._update_metric()

        scores = self.running_metric.get_scores()

        F1 = scores['F1_1']
        IoU = scores['iou_1']
        Pre = scores['precision_1']
        Rec = scores['recall_1']

        m = len(self.dataloader)

        if np.mod(self.batch_id, 100) == 1:
            message = (
                    'Is_training:%s.[%d,%d],running_F1:%.5f,running_IoU:%.5f,'
                    'running_Pre:%.5f,running_Rec:%.5f\n'
                    % (self.is_training, self.batch_id, m, F1, IoU, Pre, Rec)
            )
            self.logger.write(message)

        return

    def _collect_epoch_states(self):
        scores = self.running_metric.get_scores()
        np.save(os.path.join(self.checkpoint_dir, 'scores_dict.npy'), scores)

        F1 = scores['F1_1']
        IoU = scores['iou_1']
        Pre = scores['precision_1']
        Rec = scores['recall_1']

        self.logger.write(
            'Final_Change_Class_Result:'
            'F1:%.5f IoU:%.5f Pre:%.5f Rec:%.5f\n'
            % (F1, IoU, Pre, Rec)
        )

        message = ''
        for k, v in scores.items():
            message += '%s:%.5f ' % (k, v)
        self.logger.write('Original_scores_dict:%s\n' % message)
        self.logger.write('\n')

    def _clear_cache(self):
        self.running_metric.clear()

    def _forward_pass(self, batch):
        self.batch = batch
        img_in1 = batch['A'].to(self.device)
        img_in2 = batch['B'].to(self.device)


        outputs = self.net_G(img_in1, img_in2)
        self.G_pred = outputs[0]


    def eval_models(self, checkpoint_name='best_ckpt.pt'):

        self._load_checkpoint(checkpoint_name)

        ################## Eval ##################
        ##########################################
        self.logger.write('Begin evaluation...\n')
        self._clear_cache()
        self.is_training = False
        self.net_G.eval()

        # Iterate over data.
        for self.batch_id, batch in enumerate(self.dataloader, 0):
            with torch.no_grad():
                self._forward_pass(batch)
            self._collect_running_batch_states()
        self._collect_epoch_states()
