import argparse
import sys

from transformers import PreTrainedTokenizerFast
from transformers import BartForConditionalGeneration


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument('--path_to_tokenizerjson', type=str, help="Path to the tokenizer json file")
    parser.add_argument('--path_to_save_dir', type=str, help="Path to the save directory json file")

    args = parser.parse_args(args)

    return args


def main(args):
    args = parse_args(args)

    # Load the tokenizer from the tokenizer.json file
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=args.path_to_tokenizerjson)

    # Load the model from the model.bin file
    model = BartForConditionalGeneration.from_pretrained(args.path_to_save_dir)

    while True:
        sentence = input("Prompt: ")
        if sentence == 'exit':
            break
        input_tokens = tokenizer(sentence, return_tensors="pt")
        output = model.generate(input_tokens["input_ids"], attention_mask=input_tokens["attention_mask"])

        decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
        print("Output: ", decoded_output)


if __name__ == "__main__":
    main(sys.argv[1:])

