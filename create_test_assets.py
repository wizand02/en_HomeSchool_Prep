import pandas as pd
import pydub
import os

scratch_dir = r"c:\Users\wizan\dev_gina\en_voca_sounds"
os.makedirs(os.path.join(scratch_dir, "test_assets"), exist_ok=True)

# 1. 엑셀 파일 생성
data = {
    "LISTENING_BOOK_NM": ["TestBook", "TestBook", "TestBook"],
    "LISTENING_UNIT_NM": ["Unit 01", "Unit 01", "Unit 01"],
    "FILENAME": ["test_01", "test_02", "test_03"],
    "SENTENCE": ["Hello, nice to meet you.", "How are you doing today?", "Thank you, goodbye."]
}
df = pd.DataFrame(data)

excel_path = os.path.join(scratch_dir, "test_assets", "test_listening.xlsx")
with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    df.to_excel(writer, sheet_name='본문', index=False)

print(f"Created Excel: {excel_path}")

# 2. mp3 파일 생성 (15초 무음)
try:
    silent_audio = pydub.AudioSegment.silent(duration=15000) # 15초
    mp3_path = os.path.join(scratch_dir, "test_assets", "test_audio.mp3")
    silent_audio.export(mp3_path, format="mp3")
    print(f"Created MP3: {mp3_path}")
except Exception as e:
    print(f"Audio creation error: {e}")
