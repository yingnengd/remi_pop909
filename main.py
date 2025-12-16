from model import PopMusicTransformer
from datetime import datetime
import os
import argparse
import pickle
from finetune import load_split_file

from huggingface_hub import login
from huggingface_hub import snapshot_download
from pathlib import Path

TOKEN = os.getenv("HF_TOKEN")
login(TOKEN)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--len", help="the generation length")
    parser.add_argument("--n", default=1, help="how many sample to generate")
    parser.add_argument("--only-melody", action="store_true")
    parser.add_argument("--prompt", help="the prompt midi path")
    parser.add_argument("--prompt-chord", help="the chord of prompt midi path")
    args = parser.parse_args()

    chkpt_name = 'REMI-chord-melody' if args.only_melody else "REMI-chord"
    n_target_bar = int(args.len)
            
    repo_id = "yingnengd/REMI-chord-melody" if args.only_melody else "yingnengd/REMI-chord"

    BASE_DIR = Path(".")
    BASE_DIR.mkdir(exist_ok=True)
            
    local_model_path = snapshot_download(
        repo_id=repo_id,
        local_dir=BASE_DIR / "REMI-chord-melody",
        token=TOKEN
    )
    print("✅ 模型下载/缓存完成:", local_model_path)


    chkpt_name = 'REMI-chord-melody' if args.only_melody else "REMI-chord"
    print("[INFO] Loading model...")

    # declare model
    model = PopMusicTransformer(
        checkpoint=chkpt_name,
        is_training=False)
    
    if args.prompt is None:
        # generate from scratch
        for _ in range(int(args.n)):
            model.generate(
                n_target_bar=n_target_bar + 1,  # including N:N
                temperature=1.2,
                topk=5,
                output_path=f"./result/gen({chkpt_name})-{n_target_bar}bar_{datetime.now().strftime('%m-%d_%H%M%S')}.mid",
                prompt_paths=None)
    else:
        # generate continuation
        prompt_paths = {
            'midi_path': args.prompt,
            'melody_annotation_path': None,
            'chord_annotation_path': args.prompt_chord,
        }
        prompt_id = args.prompt.split("/")[-1].split(".")[0].split("_")[-1]
        for _ in range(int(args.n)):
            model.generate(
                n_target_bar=n_target_bar,
                temperature=1.2,
                topk=5,
                output_path=f"./result/prompt_gen({chkpt_name})-{n_target_bar}bar-({prompt_id})_{datetime.now().strftime('%m-%d_%H%M%S')}.mid",
                prompt_paths=prompt_paths)
    
    # close model
    model.close()

if __name__ == '__main__':
    main()





