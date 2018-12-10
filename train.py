import torch
import numpy as np
import argparse
from load_model import load_model
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from clevr_data import ClevrDataLoader, load_vocab
from pathlib import Path
import torchvision.utils as vutils
from torchvision import transforms
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import matplotlib.pyplot as plt
import textwrap


def clip_grads(net):
    parameters = list(filter(lambda p: p.grad is not None, net.parameters()))
    for p in parameters:
        p.grad.data.clamp_(args.min_grad, args.max_grad)


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=str, default="PG_endtoend",
                        help='Model: SAN, SAN_wbw, PG, PG_memory, PG_endtoend', metavar='')
    parser.add_argument('--question_size', type=int, default=92,
                        help='Number of words in question dictionary', metavar='')
    parser.add_argument('--stem_dim', type=int, default=256,
                        help='Number of feature-maps ', metavar='')
    parser.add_argument('--n_channel', type=int, default=1024,
                        help='Number of features channels ', metavar='')
    parser.add_argument('--answer_size', type=int, default=31,
                        help='Number of words in answers dictionary', metavar='')
    parser.add_argument('--batch_size', type=int, default=15,
                        help='Batch size', metavar='')
    parser.add_argument('--min_grad', type=float, default=-10,
                        help='Minimum value of gradient clipping', metavar='')
    parser.add_argument('--max_grad', type=float, default=10,
                        help='Maximum value of gradient clipping', metavar='')
    parser.add_argument('--load_model_path', type=str, default='./checkpoint/PG_endtoend.combined.model',
                        help='Checkpoint path', metavar='')
    parser.add_argument('--load_model_mode', type=str, default='PG+EE',
                        help='Load model checkpoint (PG, EE, PG+EE)', metavar='')
    parser.add_argument('--save_model', type=bool, default=True,
                        help='Save model checkpoint', metavar='')
    parser.add_argument('--clevr_dataset', type=str, default='/nas/softechict/CLEVR_v1.0/data_h5/',
                        help='Clevr dataset features,questions', metavar='')
    parser.add_argument('--clevr_val_images', type=str, default='/nas/softechict/CLEVR_v1.0/images/val/',
                        help='Clevr dataset validation images path', metavar='')
    parser.add_argument('--num_iterations', type=int, default=1000,
                        help='Num iteration per epoch', metavar='')
    parser.add_argument('--num_val_samples', type=int, default=200,
                        help='Num samples from test dataset', metavar='')
    parser.add_argument('--batch_multiplier', type=int, default=1,
                        help='Virtual batch size (min: 1)', metavar='')
    parser.add_argument('--train_mode', type=str, default='PG',
                        help='Train mode (PG, EE, PG+EE)', metavar='')
    parser.add_argument('--decoder_mode', type=str, default='hard+penalty',
                        help='Seq2seq decoder mode: (soft, gumbel, hard)', metavar='')
    parser.add_argument('--use_curriculum', type=bool, default=False,
                        help='Use curriculum to learn program generator', metavar='')
    return parser.parse_args()


def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if p.requires_grad and ("bias" not in n) and (p.grad is not None):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k")
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.show()


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
        if not (question.size(0) == feats.size(0) and feats.size(0) == args.batch_size):
            continue
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
            image_path = args.clevr_val_images + "CLEVR_val_" + str(int(image[0])).zfill(6) + ".png"
            question_text = "Q: "
            answer_text = vocab['answer_idx_to_token'][int(preds[0])]
            for q in question[0]:
                if int(q) == 0:
                    break
                if int(q) not in (1, 2, 3):
                    question_text += vocab['question_idx_to_token'][int(q)] + " "
            pic = image_to_tensor(image_path, question_text + "?  A: " + answer_text)
            writer.add_image('Test Images', pic, num_samples)
            if 'SAN' in args.model:
                att_map = model.getData()
                pic1 = vutils.make_grid(att_map, normalize=True, scale_each=True)
                writer.add_image('Test Attention Map', pic1, num_samples)

        if num_samples >= args.num_val_samples or num_samples == len(val_loader):
            break

    accuracy = float(int(num_correct) / num_samples)
    model.train()
    return accuracy


def train_loop(model, train_loader, val_loader, vocab):

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer_model, optimizer_pg = None, None

    if 'endtoend'in args.model:
        optimizer_pg = torch.optim.Adam([param for name, param in model.named_parameters()
                                         if 'program_generator' in name], lr=1e-04)
        optimizer_model = torch.optim.Adam([param for name, param in model.named_parameters()
                                            if 'program_generator' not in name], lr=1e-04)
    else:
        optimizer_model = torch.optim.Adam(model.parameters(), lr=1e-04)

    best_accuracy = float(0)
    t, epoch = 0, 0
    raw_reward, raw_penalty, reward_moving_average, penalty_moving_average = 0, 0, 0, 0
    loss, rew_mean, entropy_a, entropy_b = 0, 0, 0, 0
    centered_reward, centered_penalty = 0, 0
    q_fun = 0

    accuracy = check_accuracy(val_loader, model, vocab)
    writer.add_scalar('Test Accuracy', accuracy, 0)

    while epoch < 1000:
        print('------- Starting epoch %d ---------' % epoch)
        torch.set_grad_enabled(True)
        count = 1
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

            if 'hard' not in args.decoder_mode:
                loss = criterion(outs, answers.to(device)) / args.batch_multiplier
                loss.backward()

            # Optimizer virtual-batch
            count -= 1
            if count == 0:
                if 'PG' in args.train_mode:
                    if 'hard' in args.decoder_mode:
                        optimizer_pg.zero_grad()

                        if 'penalty' in args.decoder_mode:
                            # Penalty = Cross entropy loss(Y, Y_pred)
                            raw_penalty = F.nll_loss(F.log_softmax(outs.to(device).data, -1), answers.to(device).data,
                                                     reduction='none').neg()

                            # Penalty Baseline
                            penalty_moving_average *= 0.9
                            penalty_moving_average += (1.0 - 0.9) * raw_penalty.mean()
                            centered_penalty = (raw_penalty - penalty_moving_average).to(device)

                        # Reward Baseline
                        raw_reward = (preds == answers).float()
                        reward_moving_average *= 0.9
                        reward_moving_average += (1.0 - 0.9) * raw_reward.mean()
                        centered_reward = (raw_reward - reward_moving_average).to(device)

                        # REINFORCE
                        if 'penalty' in args.decoder_mode:
                            loss, rew_mean, entropy_a, entropy_b = model.program_generator.reinforce_penalty(centered_reward, centered_penalty)
                        else:
                            loss, rew_mean, entropy_a, entropy_b = model.program_generator.reinforce_reward(centered_reward)

                        # GRADIENT DEBUG
                        # plot_grad_flow(model.named_parameters())
                        optimizer_pg.step()
                    else:
                        optimizer_pg.step()
                        optimizer_pg.zero_grad()
                if 'EE' in args.train_mode:
                    optimizer_model.step()
                    optimizer_model.zero_grad()
                count = args.batch_multiplier

            # Log on Tensorboard
            if t % 5 == 0:
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

                    if 'hard' not in args.decoder_mode:
                        print("Loss: ", loss.item() * args.batch_multiplier)
                        writer.add_scalar('Train Loss', loss.item() * args.batch_multiplier,
                                          t + epoch * args.num_iterations)
                    else:
                        # print("Norm Reward AVG: ", raw_reward.mean() * args.batch_multiplier)
                        writer.add_scalar('Norm Reward AVG', raw_reward.mean() * args.batch_multiplier,
                                          t + epoch * args.num_iterations)
                        # print("Policy Loss: ", loss.item() * args.batch_multiplier)
                        writer.add_scalar('Policy Loss', loss.item() * args.batch_multiplier,
                                          t + epoch * args.num_iterations)
                        if 'penalty' in args.decoder_mode:
                            writer.add_scalar('Cross entropy Loss', raw_penalty.mean() * args.batch_multiplier,
                                              t + epoch * args.num_iterations)
                        writer.add_scalar('Entropy_a', entropy_a.item() * args.batch_multiplier,
                                          t + epoch * args.num_iterations)
                        writer.add_scalar('Entropy_b', entropy_b.item() * args.batch_multiplier,
                                          t + epoch * args.num_iterations)

            # Check accuracy on test dataset
            if t >= args.num_iterations:
                t = 0
                epoch += 1
                accuracy = check_accuracy(val_loader, model, vocab)
                writer.add_scalar('Test Accuracy', accuracy, epoch)
                if accuracy >= 0 and args.save_model:
                    best_accuracy = accuracy
                    torch.save(model.state_dict(), './checkpoint/' + args.model + '.model')
                break
            t += 1


def main():

    print("--------- Load vocab -----------")
    vocab = load_vocab('/homes/rdicarlo/scripts/vocab.json')

    train_loader_kwargs = {
        # /nas/softechict/CLEVR_v1.0/data_h5/   D://VQA//data
        'question_h5': Path(args.clevr_dataset + 'train_questions.h5'),
        'feature_h5': Path(args.clevr_dataset + 'train_features.h5'),
        'batch_size': args.batch_size,
        'num_workers': 0,
        'shuffle': True
    }

    val_loader_kwargs = {
        'question_h5': Path(args.clevr_dataset + 'val_questions.h5'),
        'feature_h5': Path(args.clevr_dataset + 'val_features.h5'),
        'batch_size': args.batch_size,
        'num_workers': 0,
        'shuffle': True
    }

    model = load_model(args, vocab)
    print(model)

    print("--------- Number of parameters -----------")
    print(model.calculate_num_params())

    print("--------- Loading checkpoint -----------")
    model = load_checkpoint(model)

    print("--------- Start training -----------")
    with ClevrDataLoader(**train_loader_kwargs) as train_loader, ClevrDataLoader(**val_loader_kwargs) as val_loader:
        train_loop(model, train_loader, val_loader, vocab)


def load_checkpoint(model):
    # Load checkpoint, partial or full
    if args.load_model_mode != '':
        model_dict = model.state_dict()
        print(args.load_model_path)
        checkpoint = torch.load(args.load_model_path)
        for key, val in checkpoint.copy().items():
            if 'program_generator' in key and args.load_model_mode == 'EE':
                checkpoint.pop(key, None)
            elif 'program_generator' not in key and args.load_model_mode == 'PG':
                checkpoint.pop(key, None)
        model_dict.update(checkpoint)
        model.load_state_dict(model_dict)
    return model


if __name__ == "__main__":
    args = parse_arguments()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    writer = SummaryWriter()
    main()
