# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import List, Optional

import fire

from llama import Llama, Dialog


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
):

    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    dialogs = [
        []
    ]
    dialogs[0].append({"role": "system", "content": "Provide short answers like an Skyrim NPC named Elias"})


    while True:
        user_input = input("whats up? ")

        if user_input == 'exit':
            print ('Goodbye')
            break

        dialogs[0].append({"role": "user", "content": user_input})

        results = generator.chat_completion(
            dialogs,  # type: ignore
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        dialogs[0].append(results[0]['generation'])
        print('Elias: ' + results[0]['generation']['content'])



if __name__ == "__main__":
    fire.Fire(main)
