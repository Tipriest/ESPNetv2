import loadData as ld
import os
import torch
import pickle
from cnn import SegmentationModel as net
import torch.backends.cudnn as cudnn
import Transforms as myTransforms
import DataSet as myDataLoader
from argparse import ArgumentParser
from train_utils import train, val, netParams, save_checkpoint, poly_lr_scheduler
import torch.optim.lr_scheduler


def reset_cuda_mem():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def print_cuda_mem(tag=""):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        alloc = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        peak = torch.cuda.max_memory_allocated() / 1024**2
        print(
            f"[CUDA Mem] {tag} | "
            f"allocated={alloc:.1f} MB, reserved={reserved:.1f} MB, peak={peak:.1f} MB"
        )


#============================================
__author__ = "Sachin Mehta"
__license__ = "MIT"
__maintainer__ = "Sachin Mehta"
#============================================

def _build_ctem_test_lists(data_dir):
    list_path = os.path.join(data_dir, 'TestSet', 'ImageSets', 'test.txt')
    test_root = os.path.join(data_dir, 'TestSet')
    im_list = []
    label_list = []
    with open(list_path, 'r') as handle:
        for line in handle:
            name = line.strip()
            if not name:
                continue
            im_list.append(os.path.join(test_root, 'JPEGImages', name + '.jpg'))
            label_list.append(os.path.join(test_root, 'SegmentationClass', name + '.png'))
    return im_list, label_list


def trainValidateSegmentation(args):
    '''
    Main function for trainign and validation
    :param args: global arguments
    :return: None
    '''

    # load the model
    cuda_available = torch.cuda.is_available()
    num_gpus = torch.cuda.device_count()
    model = net.EESPNet_Seg(args.classes, s=args.s, pretrained=args.pretrained, gpus=num_gpus)

    if num_gpus >= 1:
        model = torch.nn.DataParallel(model)

    args.savedir = args.savedir + str(args.s) + '/'

    # create the directory if not exist
    if not os.path.exists(args.savedir):
        os.mkdir(args.savedir)

    # check if processed data file exists or not
    if not os.path.isfile(args.cached_data_file):
        dataLoad = ld.LoadData(args.data_dir, args.classes, args.cached_data_file,
                               ignore_label=args.ignore_label, map_ignore_to=args.map_ignore_to)
        data = dataLoad.processData()
        if data is None:
            print('Error while pickling data. Please check.')
            exit(-1)
    else:
        data = pickle.load(open(args.cached_data_file, "rb"))



    if cuda_available:
        args.onGPU = True
        model = model.cuda()

    total_paramters = netParams(model)
    print('Total network parameters: ' + str(total_paramters))

    # define optimization criteria
    weight = torch.from_numpy(data['classWeights']) # convert the numpy array to torch
    if args.onGPU:
        weight = weight.cuda()

    criteria = torch.nn.CrossEntropyLoss(weight, ignore_index=args.ignore_label)

    if args.onGPU:
        criteria = criteria.cuda()

    print('Data statistics')
    print(data['mean'], data['std'])
    print(data['classWeights'])

    #compose the data with transforms
    trainDataset_main = myTransforms.Compose([
        myTransforms.Normalize(mean=data['mean'], std=data['std']),
        myTransforms.RandomCropResize(size=(args.inWidth, args.inHeight)),
        myTransforms.RandomFlip(),
        #myTransforms.RandomCrop(64).
        myTransforms.ToTensor(args.scaleIn),
        #
    ])

    trainDataset_scale1 = myTransforms.Compose([
        myTransforms.Normalize(mean=data['mean'], std=data['std']),
        myTransforms.RandomCropResize(size=(int(args.inWidth*1.5), int(1.5*args.inHeight))),
        myTransforms.RandomFlip(),
        myTransforms.ToTensor(args.scaleIn),
        #
    ])

    trainDataset_scale2 = myTransforms.Compose([
        myTransforms.Normalize(mean=data['mean'], std=data['std']),
        myTransforms.RandomCropResize(size=(int(args.inWidth*1.25), int(1.25*args.inHeight))), # 1536, 768
        myTransforms.RandomFlip(),
        myTransforms.ToTensor(args.scaleIn),
        #
    ])

    trainDataset_scale3 = myTransforms.Compose([
        myTransforms.Normalize(mean=data['mean'], std=data['std']),
        myTransforms.RandomCropResize(size=(int(args.inWidth*0.75), int(0.75*args.inHeight))),
        myTransforms.RandomFlip(),
        myTransforms.ToTensor(args.scaleIn),
        #
    ])

    trainDataset_scale4 = myTransforms.Compose([
        myTransforms.Normalize(mean=data['mean'], std=data['std']),
        myTransforms.RandomCropResize(size=(int(args.inWidth*0.5), int(0.5*args.inHeight))),
        myTransforms.RandomFlip(),
        myTransforms.ToTensor(args.scaleIn),
        #
    ])


    valDataset = myTransforms.Compose([
        myTransforms.Normalize(mean=data['mean'], std=data['std']),
        myTransforms.Scale(args.inWidth, args.inHeight),
        myTransforms.ToTensor(args.scaleIn),
        #
    ])

    # since we training from scratch, we create data loaders at different scales
    # so that we can generate more augmented data and prevent the network from overfitting

    trainLoader = torch.utils.data.DataLoader(
        myDataLoader.MyDataset(data['trainIm'], data['trainAnnot'], transform=trainDataset_main,
                       ignore_label=args.ignore_label, map_ignore_to=args.map_ignore_to),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    trainLoader_scale1 = torch.utils.data.DataLoader(
        myDataLoader.MyDataset(data['trainIm'], data['trainAnnot'], transform=trainDataset_scale1,
                       ignore_label=args.ignore_label, map_ignore_to=args.map_ignore_to),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    trainLoader_scale2 = torch.utils.data.DataLoader(
        myDataLoader.MyDataset(data['trainIm'], data['trainAnnot'], transform=trainDataset_scale2,
                       ignore_label=args.ignore_label, map_ignore_to=args.map_ignore_to),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    trainLoader_scale3 = torch.utils.data.DataLoader(
        myDataLoader.MyDataset(data['trainIm'], data['trainAnnot'], transform=trainDataset_scale3,
                       ignore_label=args.ignore_label, map_ignore_to=args.map_ignore_to),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    trainLoader_scale4 = torch.utils.data.DataLoader(
        myDataLoader.MyDataset(data['trainIm'], data['trainAnnot'], transform=trainDataset_scale4,
                       ignore_label=args.ignore_label, map_ignore_to=args.map_ignore_to),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    valLoader = torch.utils.data.DataLoader(
        myDataLoader.MyDataset(data['valIm'], data['valAnnot'], transform=valDataset,
                       ignore_label=args.ignore_label, map_ignore_to=args.map_ignore_to),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    if args.onGPU:
        cudnn.benchmark = True

    start_epoch = 0
    best_val = 0
    lr = args.lr

    optimizer = torch.optim.Adam(model.parameters(), lr, (0.9, 0.999), eps=1e-08, weight_decay=5e-4)
    # we step the loss by 2 after step size is reached
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_loss, gamma=0.5)



    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_val = checkpoint['best_val']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.model_path:
        if os.path.isfile(args.model_path):
            print("=> loading model weights '{}'".format(args.model_path))
            state_dict = torch.load(args.model_path)
            model.load_state_dict(state_dict)
        else:
            print("=> no model weights found at '{}'".format(args.model_path))


    logFileLoc = args.savedir + args.logFile
    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a')
    else:
        logger = open(logFileLoc, 'w')
        logger.write("Parameters: %s" % (str(total_paramters)))
        logger.write("\n{}\t{}\t{}\t{}\t{}\t".format(
            'Epoch', 'Loss(Tr)', 'Loss(val)', 'mIOU (tr)', 'mIOU (val)'))
    logger.flush()



    if args.eval_only:
        if args.eval_split == 'test' and args.dataset == 'ctem':
            test_im, test_annot = _build_ctem_test_lists(args.data_dir)
            eval_loader = torch.utils.data.DataLoader(
                myDataLoader.MyDataset(test_im, test_annot, transform=valDataset,
                                       ignore_label=args.ignore_label, map_ignore_to=args.map_ignore_to),
                batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        else:
            eval_loader = valLoader

        reset_cuda_mem()
        lossVal, overall_acc_val, per_class_acc_val, per_class_iu_val, mIOU_val = val(
            args, eval_loader, model, criteria)
        print_cuda_mem(tag=f"eval:{args.eval_split}")
        print("Eval split: {}".format(args.eval_split))
        print("Loss = {:.4f} mIOU = {:.4f}".format(lossVal, mIOU_val))
        print("Per Class Acc: {}".format(per_class_acc_val))
        print("Per Class mIoU: {}".format(per_class_iu_val))
        logger.close()
        return

    for epoch in range(start_epoch, args.max_epochs):

        #scheduler.step(epoch)
        poly_lr_scheduler(args, optimizer, epoch)
        lr = 0
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        print("Learning rate: " +  str(lr))

        # train for one epoch
        # We consider 1 epoch with all the training data (at different scales)
        train(args, trainLoader_scale1, model, criteria, optimizer, epoch)
        train(args, trainLoader_scale2, model, criteria, optimizer, epoch)
        train(args, trainLoader_scale4, model, criteria, optimizer, epoch)
        train(args, trainLoader_scale3, model, criteria, optimizer, epoch)
        lossTr, overall_acc_tr, per_class_acc_tr, per_class_iu_tr, mIOU_tr = train(args, trainLoader, model, criteria, optimizer, epoch)

        # evaluate on validation set
        lossVal, overall_acc_val, per_class_acc_val, per_class_iu_val, mIOU_val = val(args, valLoader, model, criteria)

        is_best = mIOU_val > best_val
        best_val = max(mIOU_val, best_val)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr': lr,
            'best_val': best_val,
        }, args.savedir + 'checkpoint.pth.tar')

        #save the model also
        if is_best:
            model_file_name = args.savedir + os.sep + 'model_best.pth'
            torch.save(model.state_dict(), model_file_name)

        with open(args.savedir + 'acc_' + str(epoch) + '.txt', 'w') as log:
            log.write("\nEpoch: %d\t Overall Acc (Tr): %.4f\t Overall Acc (Val): %.4f\t mIOU (Tr): %.4f\t mIOU (Val): %.4f" % (epoch, overall_acc_tr, overall_acc_val, mIOU_tr, mIOU_val))
            log.write('\n')
            log.write('Per Class Training Acc: ' + str(per_class_acc_tr))
            log.write('\n')
            log.write('Per Class Validation Acc: ' + str(per_class_acc_val))
            log.write('\n')
            log.write('Per Class Training mIOU: ' + str(per_class_iu_tr))
            log.write('\n')
            log.write('Per Class Validation mIOU: ' + str(per_class_iu_val))

        logger.write("\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.7f" % (epoch, lossTr, lossVal, mIOU_tr, mIOU_val, lr))
        logger.flush()
        print("Epoch : " + str(epoch) + ' Details')
        print("\nEpoch No.: %d\tTrain Loss = %.4f\tVal Loss = %.4f\t mIOU(tr) = %.4f\t mIOU(val) = %.4f" % (epoch, lossTr, lossVal, mIOU_tr, mIOU_val))
    logger.close()


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--model', default="ESPNetv2", help='Model name')
    parser.add_argument('--dataset', default="cityscapes", help='dataset: cityscapes or ctem')
    parser.add_argument('--data_dir', default="./city", help='Data directory')
    parser.add_argument('--inWidth', type=int, default=1024, help='Width of RGB image')
    parser.add_argument('--inHeight', type=int, default=512, help='Height of RGB image')
    parser.add_argument('--scaleIn', type=int, default=1, help='For ESPNet-C, scaleIn=8. For ESPNet, scaleIn=1')
    parser.add_argument('--max_epochs', type=int, default=300, help='Max. number of epochs')
    parser.add_argument('--num_workers', type=int, default=12, help='No. of parallel threads')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size. 12 for ESPNet-C and 6 for ESPNet. '
                                                                   'Change as per the GPU memory')
    parser.add_argument('--step_loss', type=int, default=100, help='Decrease learning rate after how many epochs.')
    parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate')
    parser.add_argument('--savedir', default='./results_espnetv2_', help='directory to save the results')
    parser.add_argument('--resume', type=str, default='', help='Use this flag to load last checkpoint for training')
    parser.add_argument('--classes', type=int, default=20, help='No of classes in the dataset. 20 for cityscapes')
    parser.add_argument('--cached_data_file', default='city.p', help='Cached file name')
    parser.add_argument('--ignore_label', type=int, default=255, help='Ignore label in ground truth masks')
    parser.add_argument('--map_ignore_to', type=int, default=19,
                        help='Map ignore label to class index (set to -1 to disable mapping)')
    parser.add_argument('--logFile', default='trainValLog.txt', help='File that stores the training and validation logs')
    parser.add_argument('--pretrained', default='', help='Pretrained ESPNetv2 weights.')
    parser.add_argument('--s', default=1, type=float, help='scaling parameter')
    parser.add_argument('--eval_only', action='store_true', help='Only run evaluation (no training)')
    parser.add_argument('--eval_split', default='val', choices=['val', 'test'], help='Split to evaluate')
    parser.add_argument('--model_path', default='', help='Path to model weights for evaluation')

    args = parser.parse_args()

    if args.map_ignore_to < 0:
        args.map_ignore_to = None

    if args.dataset == 'ctem':
        args.classes = 6
        args.inWidth = 960
        args.inHeight = 540
        args.cached_data_file = 'ctem.p'
        args.map_ignore_to = None

    trainValidateSegmentation(args)

