import torch
import numpy as np
import argparse
from load_model import load_model
from tensorboardX import SummaryWriter
from clevr_data import ClevrDataLoader, load_vocab
from pathlib import Path
import torchvision.utils as vutils
from torchvision import transforms
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import textwrap


def clip_grads(net):
    parameters = list(filter(lambda p: p.grad is not None, net.parameters()))
    for p in parameters:
        p.grad.data.clamp_(args.min_grad, args.max_grad)


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=str, default="PG_memory",
                        help='Model: SAN, SAN_wbw, PG, PG_memory', metavar='')
    parser.add_argument('--question_size', type=int, default=92,
                        help='Number of words in question dictionary', metavar='')
    parser.add_argument('--stem_dim', type=int, default=256,
                        help='Number of feature-maps ', metavar='')
    parser.add_argument('--n_channel', type=int, default=1024,
                        help='Number of features channels ', metavar='')
    parser.add_argument('--answer_size', type=int, default=31,
                        help='Number of words in answers dictionary', metavar='')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size', metavar='')
    parser.add_argument('--min_grad', type=float, default=-10,
                        help='Minimum value of gradient clipping', metavar='')
    parser.add_argument('--max_grad', type=float, default=10,
                        help='Maximum value of gradient clipping', metavar='')
    parser.add_argument('--loadmodel', type=bool, default=False,
                        help='Load model checkpoint', metavar='')
    parser.add_argument('--savemodel', type=bool, default=True,
                        help='Save model checkpoint', metavar='')
    parser.add_argument('--num_iterations', type=int, default=1000,
                        help='Num iteration per epoch', metavar='')
    parser.add_argument('--num_val_samples', type=int, default=1000,
                        help='Num samples from test dataset', metavar='')
    parser.add_argument('--batch_multiplier', type=int, default=1,
                        help='Virtual batch size (min: 1)', metavar='')

    return parser.parse_args()


def image_to_tensor(image_path, text):
    '''
    Load image file, add text, convert to tensor
    '''
    img1 = Image.open(image_path)
    img1 = img1.crop((0, 0, 450, 500))
    draw = ImageDraw.Draw(img1)
    draw.rectangle((0, 320, 450, 500), fill="white")
    font = ImageFont.truetype('font/Roboto-Black.ttf', size=22)
    lines = textwrap.wrap(text, width=40)
    y_text = 320
    for line in lines:
        width, height = font.getsize(line)
        draw.text(((450 - width) / 2, y_text), line, font=font, fill='black')
        y_text += height

    content_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    pic_tensor = content_transform(img1)
    return pic_tensor


def check_accuracy(val_loader, model, vocab):
    num_correct, num_samples = 0, 0
    model.eval()
    torch.set_grad_enabled(False)

    for question, image, feats, answers, programs in val_loader:
        feats = feats.to(device)
        question = question.to(device)
        programs = programs.to(device)

        if 'SAN' in args.model or 'endtoend' in args.model:
            outs = model(feats, question)
        else:
            outs = model(feats, programs)

        _, preds = outs.data.cpu().max(1)
        num_correct += (preds.to('cpu') == answers).sum()
        num_samples += preds.size(0)

        if num_samples % args.batch_size*3 == 0:
            image_path = "/nas/softechict/CLEVR_v1.0/images/val/CLEVR_val_" + str(int(image[0])).zfill(6) + ".png"
            question_text = "Q: "
            answer_text = vocab['answer_idx_to_token'][int(preds[0])]
            for q in question[0]:
                if int(q) == 0:
                    break
                if int(q) not in (1, 2, 3):
                    question_text += vocab['question_idx_to_token'][int(q)] + " "
            pic = image_to_tensor(image_path, question_text + "?  A: " + answer_text)
            writer.add_image('Test Images', pic, num_samples)

        if num_samples >= args.num_val_samples or num_samples == len(val_loader):
            break

    accuracy = float(int(num_correct) / num_samples)
    model.train()
    return accuracy


def train_loop(model, train_loader, val_loader, vocab):

    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), 1e-04)
    best_accuracy = float(0)
    epoch = 0
    t = 0

    accuracy = check_accuracy(val_loader, model, vocab)
    writer.add_scalar('Test Accuracy', accuracy, 0)

    while epoch < 1000:
        print('Starting epoch %d' % epoch)
        torch.set_grad_enabled(True)

        for question, _, feats, answers, programs in train_loader:
            # Check batch_size
            if not (question.size(0) == feats.size(0) and feats.size(0) == args.batch_size):
                continue

            feats = feats.to(device)
            question = question.to(device)
            programs = programs.to(device)

            if 'SAN' in args.model or 'endtoend'in args.model:
                outs = model(feats, question)
            else:
                outs = model(feats, programs)

            _, preds = outs.data.cpu().max(1)

            loss = criterion(outs, answers.to(device))
            loss.backward()
            clip_grads(model)
            optimizer.step()
            optimizer.zero_grad()

            if t % 5 == 0:
                print("Loss: ", loss.item())
                writer.add_scalar('Train Loss', loss.item(), t+epoch*args.num_iterations)

                if 'SAN' in args.model:
                    att_map = model.getData()
                    pic1 = vutils.make_grid(att_map, normalize=True, scale_each=True)
                    writer.add_image('Attention Map', pic1, t+epoch*args.num_iterations)
                elif "memory" in args.model or "endtoend" in args.model:
                    addr_u, addr_b = model.getData()
                    if addr_u:
                        pic1 = vutils.make_grid(addr_u, normalize=True, scale_each=True)
                        writer.add_image('Addressing unary', pic1, t + epoch * args.num_iterations)
                    if addr_b:
                        pic2 = vutils.make_grid(addr_b, normalize=True, scale_each=True)
                        writer.add_image('Addressing binary', pic2, t + epoch * args.num_iterations)

            if t >= args.num_iterations:
                t = 0
                epoch += 1
                accuracy = check_accuracy(val_loader, model, vocab)
                writer.add_scalar('Test Accuracy', accuracy, epoch)
                if accuracy >= best_accuracy and args.savemodel:
                    best_accuracy = accuracy
                    torch.save(model.state_dict(), './checkpoint/' + args.model + '.model')
                break
            t += 1


def main():

    vocab = load_vocab('/homes/rdicarlo/scripts/vocab.json')

    train_loader_kwargs = {
        # /nas/softechict/CLEVR_v1.0/data_h5/   D://VQA//data
        'question_h5': Path('/nas/softechict/CLEVR_v1.0/data_h5/train_questions.h5'),
        'feature_h5': Path('/nas/softechict/CLEVR_v1.0/data_h5/train_features.h5'),
        'batch_size': args.batch_size,
        'num_workers': 0,
        'shuffle': True
    }

    val_loader_kwargs = {
        'question_h5': Path('/nas/softechict/CLEVR_v1.0/data_h5/val_questions.h5'),
        'feature_h5': Path('/nas/softechict/CLEVR_v1.0/data_h5/val_features.h5'),
        'batch_size': args.batch_size,
        'num_workers': 0,
        'shuffle': True
    }

    model = load_model(args, vocab)
    print(model)

    print("--------- Number of parameters -----------")
    print(model.calculate_num_params())
    print("--------- Start training -----------")

    if args.loadmodel:
        model.load_state_dict(torch.load('./checkpoint/' + args.model + '.model'))

    with ClevrDataLoader(**train_loader_kwargs) as train_loader, ClevrDataLoader(**val_loader_kwargs) as val_loader:
        train_loop(model, train_loader, val_loader, vocab)


if __name__ == "__main__":
    args = parse_arguments()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    writer = SummaryWriter()
    main()
