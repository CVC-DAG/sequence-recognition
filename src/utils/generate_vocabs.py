import json
import re

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List, Tuple

RE_SEPARATOR = re.compile(r"[ \n]")
RE_CIPHER = re.compile(r"(\w+)_.*")
root_path = Path("/home/ptorras/Documents/Datasets/decrypt/Validated")


def produce_page_tokens(
        page_path: Path
) -> List[str]:
    with open(page_path, 'r', encoding="ISO-8859-1") as f_txt:
        txt = f_txt.read()
    page_toks = list(set(RE_SEPARATOR.split(txt)))
    print(page_toks)
    return page_toks


def get_page_paths(
        root_path: Path
) -> Tuple[List[Path], List[str]]:

    out_paths = []
    out_dsets = []

    for folder in root_path.iterdir():
        matches = RE_CIPHER.match(folder.name)
        cipher_name = matches.group(1)

        for page in folder.iterdir():
            txtfile = list(page.glob("*.txt"))[0]
            out_paths.append(txtfile)
            out_dsets.append(cipher_name)

    return out_paths, out_dsets


#%%
RE_PLAINTEXT = re.compile("")


#%%
paths, ciphers = get_page_paths(root_path)

tok_vocabs = {cipher: [] for cipher in list(set(ciphers))}

for file, cipher in zip(paths, ciphers):
    tok_vocabs[cipher] = list(set(
        tok_vocabs[cipher] + produce_page_tokens(file)
    ))

tok_vocabs.items()
