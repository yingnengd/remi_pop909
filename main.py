'''
from model import PopMusicTransformer
from datetime import datetime
import os
import argparse
import pickle
from finetune import load_split_file

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--len", help="the generation length")
    parser.add_argument("-n", default=1, help="how many sample to generate")
    parser.add_argument("--only-melody", action="store_true")
    parser.add_argument("--prompt", help="the prompt midi path")
    parser.add_argument("--prompt-chord", help="the chord of prompt midi path")
    args = parser.parse_args()

    chkpt_name = 'REMI-chord-melody' if args.only_melody else "REMI-chord"
    n_target_bar = int(args.len)

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
    model.close()‘’‘

if __name__ == '__main__':
    main()
'''
#==============================
from model import PopMusicTransformer
from datetime import datetime
import os
import argparse
import pickle
from finetune import load_split_file

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# =========================
# 歌曲结构（工业级）
# =========================
SONG_STRUCTURE = [
    ("INTRO", 8, 1.0),
    ("VERSE", 16, 1.05),
    ("PRE", 8, 1.1),
    ("CHORUS", 16, 1.15),
    ("VERSE2", 16, 1.05),
    ("PRE2", 8, 1.1),
    ("CHORUS2", 16, 1.15),
    ("BRIDGE", 8, 1.2),
    ("FINAL_CHORUS", 24, 1.15),
    ("OUTRO", 8, 1.0),
]

OUTPUT_DIR = "./result/full_song"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    chkpt_name = "REMI-chord"  # 或 REMI-chord-melody

    print("[INFO] Loading model...")
    model = PopMusicTransformer(
        checkpoint=chkpt_name,
        is_training=False
    )
    print("[INFO] Model loaded.\n")

    last_midi = None
    section_midis = []

    # =========================
    # 核心：prompt 接力生成
    # =========================
    for idx, (section, bars, temperature) in enumerate(SONG_STRUCTURE):
        print(f"[GEN] {section} ({bars} bars)")

        out_midi = os.path.join(
            OUTPUT_DIR,
            f"{idx:02d}_{section}_{bars}bars_{datetime.now().strftime('%H%M%S')}.mid"
        )

        prompt_paths = None
        if last_midi is not None:
            prompt_paths = {
                "midi_path": last_midi,
                "melody_annotation_path": None,
                "chord_annotation_path": None,
            }

        model.generate(
            n_target_bar=bars,
            temperature=temperature,
            topk=5,
            output_path=out_midi,
            prompt_paths=prompt_paths
        )

        last_midi = out_midi
        section_midis.append(out_midi)

    model.close()

    print("\n🎉 FULL SONG GENERATED")
    for m in section_midis:
        print(m)


if __name__ == "__main__":
    main()




