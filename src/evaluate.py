# Deep learning libraries
import torch

# Other libraries
from typing import Final
import os
from collections import defaultdict
from tqdm import tqdm
import json

# Own modules
from src.utils import (set_seed,
                       load_data,
                       load_checkpoint,
                       save_image,
                       calculate_bleu,
                       calculate_cider)
from src.model import ImageCaptioningModel


# Set device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
set_seed(42)

# Static variables
DATA_PATH: Final[str] = "data"

# Other global variables
debug = False  # If true it will print the captions generated
beam = True  # The captions will be generated using beam search


def main() -> None:
    """
    This function is the main program. It loads the data, the model,
    and evaluates it.
    """

    dataset_name = "flickr30k"  # "flickr8k" or "flickr30k"
    captions_dir = "captions"
    solution_dir = "solution"

    # Define hyperparameters
    batch_size = 1
    embedding_size = 300
    hidden_size = 256
    num_layers = 1
    GENERATE_CAPTIONS = True

    # load data
    (_, _, test_loader, vocab) = load_data(DATA_PATH,
                                           dataset_name,
                                           batch_size,
                                           num_workers=0)

    # model = MyModel(encoder_params, decoder_params)
    model_type = ImageCaptioningModel(embedding_size,
                                      hidden_size,
                                      len(vocab),
                                      num_layers)

    _, model, _ = load_checkpoint(model_type, None, "checkpoint")

    # If the captions are not generated, generate them
    if not os.path.exists(captions_dir):
        GENERATE_CAPTIONS = True
        os.makedirs(captions_dir)

    if GENERATE_CAPTIONS:

        if not os.path.exists(solution_dir):
            os.makedirs(solution_dir)

        model = model.to(device)
        model.eval()

        # evaluate model
        with torch.no_grad():

            batch_idx = 0

            refs = defaultdict(list)
            hypos = defaultdict(list)

            for img_name, inputs, targets in tqdm(test_loader):

                inputs = inputs.to(device)
                targets = targets.to(device)

                # Inputs must be float
                inputs = inputs.float()
                targets = targets.long()

                targets = targets.squeeze(1)
                real_caption = vocab.indices_to_caption(targets.tolist())

                img_id = img_name[0]
                refs[img_id].append(real_caption)

                # Only generate the caption ones for the five images
                # that are the same
                if batch_idx % 5 == 0:

                    # Compute both the caption and the caption using beam search
                    # to compare them
                    if debug:
                        caption = model.generate_caption(inputs, vocab)
                        caption_beam = model.generate_caption_beam_search(inputs, vocab)
                        print("\nCaption: ", caption)
                        print("Caption beam search: ", caption_beam)

                        if beam:
                            caption = caption_beam

                    else:
                        # Add the caption to the hypos
                        if beam:
                            caption = model.generate_caption_beam_search(inputs, vocab)
                        else:
                            caption = model.generate_caption(inputs, vocab)

                    hypos[img_id].append(caption)
                    save_image(inputs, caption, real_caption, solution_dir, batch_idx)

                # We save the jsons every 100 images
                if batch_idx % 100 == 0:
                    with open(f"{captions_dir}/hypo.json", "w") as f:
                        json.dump(hypos, f, indent=4)

                    with open(f"{captions_dir}/refs.json", "w") as f:
                        json.dump(refs, f, indent=4)

                batch_idx += 1

        # Compute metrics
        average_bleu_score = calculate_bleu(refs, hypos)
        cider_score = calculate_cider(refs, hypos)

        print(f"Average BLEU score: {average_bleu_score:.4f}")
        print(f"CIDEr score: {cider_score:.4f}")
        print("Evaluation finished.")

        # Save the hypo and refs for the captions
        # as jsons

        with open(f"{captions_dir}/hypo.json", "w") as f:
            json.dump(hypos, f, indent=4)

        with open(f"{captions_dir}/refs.json", "w") as f:
            json.dump(refs, f, indent=4)

    else:

        # Load the hypo and refs for the captions
        # as jsons and compute BLEU and CIDEr scores
        # Only if the captions were already generated

        with open(f"{captions_dir}/hypo.json", "r") as f:
            hypos = json.load(f)

        with open(f"{captions_dir}/refs.json", "r") as f:
            refs = json.load(f)

        bleu_score = calculate_bleu(refs, hypos)
        cider_score = calculate_cider(refs, hypos)

        print(f"BLEU score: {bleu_score:.4f}")
        print(f"CIDEr score: {cider_score:.4f}")


if __name__ == "__main__":
    main()
