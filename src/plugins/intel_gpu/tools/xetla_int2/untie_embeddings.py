"""Untie a Bonsai checkpoint's output projection from its embedding.

Bonsai 1.7B and 4B set tie_word_embeddings, so lm_head aliases embed_tokens and
the exported IR has no lm_head weight of its own to compress. The embedding is
itself ternary, so giving lm_head its own copy lets it be packed like every
other projection, matching the already-untied 8B.
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    args = ap.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.src, dtype=torch.bfloat16)
    if not model.config.tie_word_embeddings:
        print("already untied; copying through")
    else:
        emb = model.get_input_embeddings().weight
        # A plain assignment would keep the alias and save_pretrained would drop it.
        model.lm_head.weight = torch.nn.Parameter(emb.detach().clone())
        model.config.tie_word_embeddings = False
        print(f"untied lm_head from embedding, shape {tuple(emb.shape)}")

    model.save_pretrained(args.dst)
    AutoTokenizer.from_pretrained(args.src).save_pretrained(args.dst)
    print("saved", args.dst)


if __name__ == "__main__":
    main()
