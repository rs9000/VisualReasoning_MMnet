import torch
import numpy as np
import argparse
from load_model import load_model
from tensorboardX import SummaryWriter
from clevr_data import get_loader, load_vocab
from pathlib import Path
import torchvision.utils as vutils


def clip_grads(net, args):
    parameters = list(filter(lambda p: p.grad is not None, net.parameters()))
    for p in parameters:
        p.grad.data.clamp_(args.min_grad, args.max_grad)


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=str, default="PG_endtoend",
                        help='Model: SAN, SAN_wbw, PG, PG_memory', metavar='')
    parser.add_argument('--fmaps', type=int, default=14*14,
                        help='Number of features in res-net maps', metavar='')
    parser.add_argument('--question_size', type=int, default=92,
                        help='Number of words in question dictionary', metavar='')
    parser.add_argument('--stem_dim', type=int, default=256,
                        help='Number of feature-maps ', metavar='')
    parser.add_argument('--answer_size', type=int, default=31,
                        help='Number of words in answers dictionary', metavar='')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size', metavar='')
    parser.add_argument('--min_grad', type=float, default=-10,
                        help='Minimum value of gradient clipping', metavar='')
    parser.add_argument('--max_grad', type=float, default=10,
                        help='Maximum value of gradient clipping', metavar='')
    parser.add_argument('--loadmodel', type=str, default='./checkpoint/PG_memory.model',
                        help='The pre-trained model checkpoint', metavar='')
    parser.add_argument('--savemodel', type=str, default='./checkpoint/PG_memory.model',
                        help='Name/Path of model checkpoint', metavar='')

    return parser.parse_args()


def train_loop(dataloader, args):

    num_correct, num_samples = 0, 0
    epoch = 0

    while epoch < 30:
        print('Starting epoch %d' % epoch)
        torch.set_grad_enabled(True)
        for t, (question, image, feats, answers, programs) in enumerate(dataloader):
            t += epoch*len(dataloader.dataset)
            losses = []

            # Check batch_size
            if not (question.size(0) == feats.size(0) and feats.size(0) == args.batch_size):
                continue

            feats = feats.to(device)
            question = question.to(device)
            programs = programs.to(device)
            optimizer.zero_grad()

            if 'SAN' or 'endtoend'in args.model:
                outs = model(feats, question)
            else:
                outs = model(feats, programs)

            _, preds = outs.max(1)
            num_samples += preds.size(0)

            loss = criterion(outs, answers.to(device))
            loss.backward()
            clip_grads(model, args)
            optimizer.step()
            losses += [loss.item()]

            if t % 2 == 0:
                mean_loss = np.array(losses).mean()
                print("Loss: ", loss.item())
                writer.add_scalar('Mean loss', mean_loss, t)
                losses = []

                if 'SAN' in args.model:
                    att_map = model.getData()
                    pic1 = vutils.make_grid(att_map, normalize=True, scale_each=True)
                    writer.add_image('Attention Map', pic1, t)
                else:
                    _, addr = model.getData()
                    pic1 = vutils.make_grid(addr, normalize=True, scale_each=True)
                    writer.add_image('Addressing', pic1, t)

            if t % 100 == 0:
                torch.save(model.state_dict(), args.savemodel)
        epoch += 1


if __name__ == "__main__":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args = parse_arguments()
    writer = SummaryWriter()

    vocab = load_vocab('D://VQA//data//vocab.json')

    train_loader_kwargs = {
        # /nas/softechict/CLEVR_v1.0/data_h5/   D://VQA//data
        'question_h5': Path('D://VQA//data//train_questions.h5'),
        'feature_h5': Path('D://VQA//data//train_features.h5'),
        'batch_size': args.batch_size,
        'num_workers': 0,
        'shuffle': True
    }

    model = load_model(args, vocab)
    print(model)

    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), 1e-04)

    print("--------- Number of parameters -----------")
    print(model.calculate_num_params())
    print("--------- Start training -----------")

    check_p = Path(args.loadmodel)
    #if check_p.is_file():
     #   model.load_state_dict(torch.load(args.loadmodel))

    dataloader = get_loader(**train_loader_kwargs)
    train_loop(dataloader, args)
