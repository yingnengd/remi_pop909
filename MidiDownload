import os
import zipfile
from IPython.display import FileLink

# 1️⃣ 设置 MIDI 文件目录
midi_dir = "/remi_pop909/result/full_song/"
zip_path = "/remi_pop909/midis.zip"

# 2️⃣ 创建 ZIP 文件
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(midi_dir):
        for file in files:
            if file.endswith(".mid") or file.endswith(".midi"):
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, midi_dir)  # 压缩包内部路径
                zipf.write(file_path, arcname)
                print(f"Added {file_path} -> {arcname}")

print(f"\n✅ All MIDI files zipped to: {zip_path}")

# 3️⃣ 生成可点击下载链接
FileLink(zip_path)
