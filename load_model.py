from models.SAN import SAN as SAN
from models.SAN_wbw import SAN as SAN_wbw
from models.PG import PG
from models.PG_memory import PG as PG_memory
from models.PG_endtoend import PG as PG_endtoend


def load_model(args, vocab):

    modelname = args.model
    model = None

    if modelname == "SAN":
        model = SAN(vocab=vocab,
                    stem_dim=args.stem_dim,
                    question_size=args.question_size,
                    n_answers=args.answer_size,
                    batch_size=args.batch_size,
                    n_channel=args.n_channel
                    ).cuda()

    elif modelname == "SAN_wbw":
        model = SAN_wbw(vocab=vocab,
                        stem_dim=args.stem_dim,
                        question_size=args.question_size,
                        n_answers=args.answer_size,
                        batch_size=args.batch_size,
                        n_channel=args.n_channel
                        ).cuda()

    elif modelname == "PG":
        model = PG(vocab=vocab,
                   stem_dim=args.stem_dim,
                   question_size=args.question_size,
                   n_answers=args.answer_size,
                   batch_size=args.batch_size,
                   n_channel=args.n_channel
                   ).cuda()

    elif modelname == "PG_memory":
        model = PG_memory(vocab=vocab,
                          stem_dim=args.stem_dim,
                          question_size=args.question_size,
                          n_answers=args.answer_size,
                          batch_size=args.batch_size,
                          n_channel=args.n_channel
                          ).cuda()

    elif modelname == "PG_endtoend":
        model = PG_endtoend(vocab=vocab,
                            stem_dim=args.stem_dim,
                            question_size=args.question_size,
                            n_answers=args.answer_size,
                            batch_size=args.batch_size,
                            n_channel=args.n_channel,
                            decoder_mode=args.decoder_mode,
                            use_curriculum=args.use_curriculum
                            ).cuda()

    return model
