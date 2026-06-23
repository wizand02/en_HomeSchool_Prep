import streamlit as st
import pandas as pd
import os
import eng_to_ipa as ipa
from gtts import gTTS
import re
import io
from pathlib import Path
import nltk

# ────────────────────────────────────────────
# 유틸리티 (수정 및 분석 영역)
# ────────────────────────────────────────────
def sanitize_filename(name):
    """파일명으로 사용할 수 없는 문자 제거"""
    name = str(name)
    return re.sub(r'[\\/*?:"<>|]', "", name).strip()


def get_update_filename(original_name: str) -> str:
    """원본 파일명에 _update를 붙인 이름 반환 (확장자 앞에 삽입)"""
    p = Path(original_name)
    return p.stem + "_update" + p.suffix


# ────────────────────────────────────────────
# 처리 함수
# ────────────────────────────────────────────
def check_typos(df):
    """단어 오탈자 점검: F열 단어가 순수 알파벳+공백인지 확인하여 의심 단어 표시"""
    st.info("🔍 단어 오탈자를 점검합니다...")
    suspect = []
    for index, row in df.iterrows():
        val_f = str(row.iloc[5]).strip() if not pd.isna(row.iloc[5]) else ""
        if val_f and not re.match(r"^[A-Za-z\s\-'\.]+$", val_f):
            suspect.append({"행 번호": index + 2, "F열 단어": val_f})

    if suspect:
        st.warning(f"⚠️ 오탈자 의심 단어 {len(suspect)}개 발견:")
        st.dataframe(pd.DataFrame(suspect))
    else:
        st.success("✅ 오탈자 의심 단어 없음 (모든 단어가 올바른 형식)")
    return df


def update_meaning(df):
    """의미 업데이트: G열에 한국어 의미를 채움 (deep-translator 사용)"""
    st.info("📖 한국어 의미를 번역하고 업데이트합니다...")
    
    try:
        from deep_translator import GoogleTranslator
        translator = GoogleTranslator(source='en', target='ko')
    except ImportError:
        st.error("deep-translator 라이브러리가 필요합니다. 'pip install deep-translator'를 실행해주세요.")
        return df

    progress_bar = st.progress(0)
    status_text = st.empty()
    total_rows = len(df)

    # G열(인덱스 6) 확보
    while len(df.columns) < 7:
        df[f"Column_{len(df.columns)}"] = ""

    count = 0
    for index, row in df.iterrows():
        val_f = str(row.iloc[5]).strip() if not pd.isna(row.iloc[5]) else ""
        if val_f:
            existing_g = str(df.iat[index, 6]).strip() if not pd.isna(df.iat[index, 6]) else ""
            # G열이 비어있거나 이전에 생성된 플레이스홀더([단어])인 경우 업데이트
            if not existing_g or existing_g == "nan" or (existing_g.startswith("[") and existing_g.endswith("]")):
                try:
                    translated = translator.translate(val_f)
                    df.iat[index, 6] = translated
                    count += 1
                except Exception as e:
                    st.warning(f"번역 오류 ({val_f}): {e}")

        progress = (index + 1) / total_rows
        progress_bar.progress(progress)
        status_text.text(f"의미 번역 중: {index + 1}/{total_rows} ({val_f})")

    st.success(f"의미 업데이트 완료: {count}개 행 처리 (한국어 번역 결과 저장)")
    return df


def update_ipa(df, base_output_dir):
    """발음기호 업데이트: H열에 IPA 발음기호 삽입"""
    st.info("🔤 발음 기호를 업데이트합니다...")
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_rows = len(df)

    # H열(인덱스 7) 확보
    while len(df.columns) < 8:
        df[f"Column_{len(df.columns)}"] = ""

    count = 0
    for index, row in df.iterrows():
        val_f = str(row.iloc[5]).strip() if not pd.isna(row.iloc[5]) else ""
        if val_f:
            pronunciation = ipa.convert(val_f)
            df.iat[index, 7] = pronunciation
            count += 1

        progress = (index + 1) / total_rows
        progress_bar.progress(progress)
        status_text.text(f"IPA 처리 중: {index + 1}/{total_rows} ({val_f})")

    st.success(f"발음기호 업데이트 완료: {count}개 행")
    return df


def update_pos(df):
    """품사 업데이트: I열에 품사 정보 삽입 (nltk 사용)"""
    st.info("🏷️ 품사를 업데이트합니다...")
    
    # nltk 데이터 준비
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        from nltk import pos_tag, word_tokenize
    except Exception as e:
        st.error(f"nltk 라이브러리 호출 중 오류 발생: {e}")
        return df

    # 품사 태그 맵핑 (간소화)
    pos_map = {
        'NN': 'n.', 'NNS': 'n.', 'NNP': 'n.', 'NNPS': 'n.',
        'VB': 'v.', 'VBD': 'v.', 'VBG': 'v.', 'VBN': 'v.', 'VBP': 'v.', 'VBZ': 'v.',
        'JJ': 'adj.', 'JJR': 'adj.', 'JJS': 'adj.',
        'RB': 'adv.', 'RBR': 'adv.', 'RBS': 'adv.',
        'IN': 'prep.',
        'PRP': 'pron.', 'PRP$': 'pron.',
        'CC': 'conj.',
        'CD': 'num.',
        'UH': 'int.'
    }

    progress_bar = st.progress(0)
    status_text = st.empty()
    total_rows = len(df)

    # I열(인덱스 8) 확보
    while len(df.columns) < 9:
        df[f"Column_{len(df.columns)}"] = ""

    count = 0
    for index, row in df.iterrows():
        val_f = str(row.iloc[5]).strip() if not pd.isna(row.iloc[5]) else ""
        if val_f:
            try:
                tokens = word_tokenize(val_f)
                tags = pos_tag(tokens)
                if tags:
                    raw_tag = tags[0][1]
                    pretty_tag = pos_map.get(raw_tag, raw_tag.lower())
                    df.iat[index, 8] = pretty_tag
                    count += 1
            except:
                pass

        progress = (index + 1) / total_rows
        progress_bar.progress(progress)
        status_text.text(f"품사 처리 중: {index + 1}/{total_rows} ({val_f})")

    st.success(f"품사 업데이트 완료: {count}개 행")
    return df


def process_reading_excel(sheets_dict):
    """리딩 파일의 모든 워크시트('본문', '단어') 처리"""
    try:
        from deep_translator import GoogleTranslator
        translator = GoogleTranslator(source='en', target='ko')
    except ImportError:
        st.error("deep-translator 라이브러리가 필요합니다. 'pip install deep-translator'를 실행해주세요.")
        return sheets_dict

    # 1. '본문' 워크시트 처리 (D열 -> E열)
    if '본문' in sheets_dict:
        st.info("📖 '본문' 시트의 해석을 업데이트합니다...")
        df = sheets_dict['본문']
        while len(df.columns) < 5:
            df[f"Column_{len(df.columns)}"] = ""
        
        progress_bar = st.progress(0)
        for index, row in df.iterrows():
            val_d = str(df.iat[index, 3]).strip() if not pd.isna(df.iat[index, 3]) else ""
            if val_d:
                # E열이 비어있는 경우에만(NaN이거나 빈 문자열) 번역 수행
                existing_e = df.iat[index, 4]
                if pd.isna(existing_e) or str(existing_e).strip() == "":
                    try:
                        df.iat[index, 4] = translator.translate(val_d)
                    except Exception as e:
                        st.warning(f"본문 번역 오류 (행 {index+2}): {e}")
            progress_bar.progress((index + 1) / len(df))
        st.success("'본문' 시트 처리 완료")

    # 2. '단어' 워크시트 처리 (C열 -> D열)
    if '단어' in sheets_dict:
        st.info("📖 '단어' 시트의 뜻을 업데이트합니다...")
        df = sheets_dict['단어']
        while len(df.columns) < 4:
            df[f"Column_{len(df.columns)}"] = ""
            
        progress_bar = st.progress(0)
        for index, row in df.iterrows():
            val_c = str(df.iat[index, 2]).strip() if not pd.isna(df.iat[index, 2]) else ""
            if val_c:
                # D열이 비어있는 경우에만 번역 수행
                existing_d = df.iat[index, 3]
                if pd.isna(existing_d) or str(existing_d).strip() == "":
                    try:
                        df.iat[index, 3] = translator.translate(val_c)
                    except Exception as e:
                        st.warning(f"단어 번역 오류 (행 {index+2}): {e}")
            progress_bar.progress((index + 1) / len(df))
        st.success("'단어' 시트 처리 완료")

    return sheets_dict


def process_listening_excel(sheets_dict):
    """리스닝 파일의 모든 워크시트('본문', '단어') 처리"""
    try:
        from deep_translator import GoogleTranslator
        translator = GoogleTranslator(source='en', target='ko')
    except ImportError:
        st.error("deep-translator 라이브러리가 필요합니다. 'pip install deep-translator'를 실행해주세요.")
        return sheets_dict

    # 1. '본문' 워크시트 처리 (D열 -> E열)
    if '본문' in sheets_dict:
        st.info("🎧 '본문' 시트의 해석을 업데이트합니다...")
        df = sheets_dict['본문']
        while len(df.columns) < 5:
            df[f"Column_{len(df.columns)}"] = ""
        
        progress_bar = st.progress(0)
        for index, row in df.iterrows():
            val_d = str(df.iat[index, 3]).strip() if not pd.isna(df.iat[index, 3]) else ""
            if val_d:
                existing_e = df.iat[index, 4]
                if pd.isna(existing_e) or str(existing_e).strip() == "":
                    # "Sally:" 와 같이 콜론으로 끝나는 텍스트는 화자를 의미하므로 그대로 복사
                    if val_d.endswith(':'):
                        df.iat[index, 4] = val_d
                    else:
                        try:
                            df.iat[index, 4] = translator.translate(val_d)
                        except Exception as e:
                            st.warning(f"본문 번역 오류 (행 {index+2}): {e}")
            progress_bar.progress((index + 1) / len(df))
        st.success("'본문' 시트 처리 완료")

    # 2. '단어' 워크시트 처리 (C열 -> D열)
    if '단어' in sheets_dict:
        st.info("📖 '단어' 시트의 뜻을 업데이트합니다...")
        df = sheets_dict['단어']
        while len(df.columns) < 4:
            df[f"Column_{len(df.columns)}"] = ""
            
        progress_bar = st.progress(0)
        for index, row in df.iterrows():
            val_c = str(df.iat[index, 2]).strip() if not pd.isna(df.iat[index, 2]) else ""
            if val_c:
                # D열이 비어있는 경우에만 번역 수행
                existing_d = df.iat[index, 3]
                if pd.isna(existing_d) or str(existing_d).strip() == "":
                    try:
                        df.iat[index, 3] = translator.translate(val_c)
                    except Exception as e:
                        st.warning(f"단어 번역 오류 (행 {index+2}): {e}")
            progress_bar.progress((index + 1) / len(df))
        st.success("'단어' 시트 처리 완료")

    return sheets_dict


def parse_listening_paragraphs(paragraphs, filename_base, current_unit, speakers=None):
    """리스닝 스크립트(문단 리스트)를 문장별로 분할하여 데이터를 정제"""
    try:
        from deep_translator import GoogleTranslator
        translator = GoogleTranslator(source='en', target='ko')
    except ImportError:
        st.error("deep-translator 라이브러리가 필요합니다. 'pip install deep-translator'를 실행해주세요.")
        return []

    # nltk sentence tokenizer 준비
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

    data = []
    sentence_no = 1

    # speakers 소문자 리스트화
    speakers_lower = [s.strip().lower() for s in speakers] if speakers else []

    # 화자가 2명 이상 등록되어 있을 때만 대화 형식(기본적으로 대화 간 공백을 생략하는 형식)으로 판단
    is_dialogue = len(speakers_lower) >= 2

    for i, p_text in enumerate(paragraphs):
        p_text = p_text.strip()
        if not p_text:
            continue

        # Unit 정보 업데이트 (예: 'Unit 01. The Sun')
        if p_text.lower().startswith("unit"):
            current_unit = p_text
            sentence_no = 1  # 단원별로 문장 번호 초기화
            continue

        # Topic 문단 여부 판별 (Topic 다음에는 항상 한 줄의 공백 추가)
        is_topic = p_text.lower().startswith("topic")

        # 대화 형식 파싱 (예: "Sally: Hello. How are you?")
        # 화자가 단독으로 오는 경우 ("Sally:")
        if p_text.endswith(":") and re.match(r"^[A-Za-z0-9\s\-]+:$", p_text):
            sound_path = f"{sanitize_filename(current_unit)}/{sanitize_filename(filename_base)}_{sentence_no}.mp3" if current_unit else f"Unassigned/{sanitize_filename(filename_base)}_{sentence_no}.mp3"
            data.append({
                "A": filename_base,
                "B": current_unit,
                "C": sentence_no,
                "D": p_text,
                "E": p_text,  # 해석하지 않고 그대로 둠
                "F": sound_path
            })
            sentence_no += 1
            
            # 문단이 끝난 후 공백 행 추가 검사
            if i < len(paragraphs) - 1:
                if is_topic or not is_dialogue:
                    data.append({
                        "A": filename_base,
                        "B": current_unit,
                        "C": sentence_no,
                        "D": "",
                        "E": "",
                        "F": ""
                    })
                    sentence_no += 1
            continue

        # 1. 수동 입력 시 화자 목록이 제공되었고, 첫 단어가 화자 목록에 있는 경우
        is_speaker_found = False
        words = p_text.split()
        if speakers_lower and words:
            first_word_raw = words[0]
            first_word_clean = re.sub(r"[^A-Za-z0-9]", "", first_word_raw).lower()
            if first_word_clean in speakers_lower:
                is_speaker_found = True
                speaker_name = first_word_raw.rstrip(":")
                body_text = p_text[len(first_word_raw):].strip().lstrip(":").strip()
                
                sentences = nltk.sent_tokenize(body_text)
                for sent in sentences:
                    sent_text = f"{speaker_name}: {sent}"
                    try:
                        meaning = translator.translate(sent_text)
                    except:
                        meaning = ""
                    
                    sound_path = f"{sanitize_filename(current_unit)}/{sanitize_filename(filename_base)}_{sentence_no}.mp3" if current_unit else f"Unassigned/{sanitize_filename(filename_base)}_{sentence_no}.mp3"
                    data.append({
                        "A": filename_base,
                        "B": current_unit,
                        "C": sentence_no,
                        "D": sent_text,
                        "E": meaning,
                        "F": sound_path
                    })
                    sentence_no += 1

        if not is_speaker_found:
            # "Sally: Hello..." 같은 형식 판별
            match = re.match(r"^([A-Za-z0-9\s\-]+:)(.*)", p_text)
            if match:
                speaker = match.group(1).strip()
                body = match.group(2).strip()
                sentences = nltk.sent_tokenize(body)
                for sent in sentences:
                    sent_text = f"{speaker} {sent}"
                    try:
                        meaning = translator.translate(sent_text)
                    except:
                        meaning = ""
                    
                    sound_path = f"{sanitize_filename(current_unit)}/{sanitize_filename(filename_base)}_{sentence_no}.mp3" if current_unit else f"Unassigned/{sanitize_filename(filename_base)}_{sentence_no}.mp3"
                    data.append({
                        "A": filename_base,
                        "B": current_unit,
                        "C": sentence_no,
                        "D": sent_text,
                        "E": meaning,
                        "F": sound_path
                    })
                    sentence_no += 1
            else:
                # 일반 서술문
                sentences = nltk.sent_tokenize(p_text)
                for sent in sentences:
                    try:
                        meaning = translator.translate(sent)
                    except:
                        meaning = ""
                    
                    sound_path = f"{sanitize_filename(current_unit)}/{sanitize_filename(filename_base)}_{sentence_no}.mp3" if current_unit else f"Unassigned/{sanitize_filename(filename_base)}_{sentence_no}.mp3"
                    data.append({
                        "A": filename_base,
                        "B": current_unit,
                        "C": sentence_no,
                        "D": sent,
                        "E": meaning,
                        "F": sound_path
                    })
                    sentence_no += 1

        # 문단이 끝난 후 공백 행 추가 검사
        if i < len(paragraphs) - 1:
            # Topic 다음이거나, 대화 형식이 아닌 경우(독백/설명문) 공백 행 추가
            if is_topic or not is_dialogue:
                data.append({
                    "A": filename_base,
                    "B": current_unit,
                    "C": sentence_no,
                    "D": "",
                    "E": "",
                    "F": ""
                })
                sentence_no += 1

    return data


def update_sound_paths(df, base_output_dir):
    """사운드파일 경로 업데이트: L열에 상대 경로 저장"""
    st.info("📂 사운드 파일 경로를 업데이트합니다...")
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_rows = len(df)

    # L열(인덱스 11) 확보
    while len(df.columns) < 12:
        df[f"Column_{len(df.columns)}"] = ""

    count = 0
    for index, row in df.iterrows():
        val_a = str(row.iloc[0]).strip() if not pd.isna(row.iloc[0]) else ""
        val_b = str(row.iloc[1]).strip() if not pd.isna(row.iloc[1]) else "Unassigned"
        val_d = str(row.iloc[3]).strip() if not pd.isna(row.iloc[3]) else ""
        val_e = str(row.iloc[4]).strip() if not pd.isna(row.iloc[4]) else ""
        val_f = str(row.iloc[5]).strip() if not pd.isna(row.iloc[5]) else ""

        if val_f:
            folder_name = sanitize_filename(val_b)
            filename_parts = [val_a, val_d, val_e, val_f]
            filename = "_".join([p for p in filename_parts if p]) + ".mp3"
            filename = sanitize_filename(filename)
            relative_path = f"{folder_name}/{filename}"

            df.iat[index, 11] = relative_path
            count += 1

        progress = (index + 1) / total_rows
        progress_bar.progress(progress)
        status_text.text(f"경로 업데이트 중: {index + 1}/{total_rows}")

    st.success(f"사운드 파일 경로 업데이트 완료: {count}개 행")
    return df


def generate_sounds(df, base_output_dir):
    """사운드 파일 생성: MP3 파일 생성 (이미 존재하면 건너뜀)"""
    st.info("🎵 사운드 파일을 생성합니다...")
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_rows = len(df)
    generated_count = 0
    skipped_count = 0

    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)

    val_f = ""
    for index, row in df.iterrows():
        try:
            val_a = str(row.iloc[0]).strip() if not pd.isna(row.iloc[0]) else ""
            val_b = str(row.iloc[1]).strip() if not pd.isna(row.iloc[1]) else "Unassigned"
            val_d = str(row.iloc[3]).strip() if not pd.isna(row.iloc[3]) else ""
            val_e = str(row.iloc[4]).strip() if not pd.isna(row.iloc[4]) else ""
            val_f = str(row.iloc[5]).strip() if not pd.isna(row.iloc[5]) else ""

            if not val_f:
                continue

            folder_name = sanitize_filename(val_b)
            target_dir = os.path.join(base_output_dir, folder_name)

            filename_parts = [val_a, val_d, val_e, val_f]
            filename = "_".join([p for p in filename_parts if p]) + ".mp3"
            filename = sanitize_filename(filename)
            file_path = os.path.join(target_dir, filename)

            if not os.path.exists(file_path):
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
                tts = gTTS(text=val_f, lang='en')
                tts.save(file_path)
                generated_count += 1
            else:
                skipped_count += 1

        except Exception as e:
            st.warning(f"Row {index+2} 처리 중 오류: {e}")

        progress = (index + 1) / total_rows
        progress_bar.progress(progress)
        status_text.text(
            f"파일 생성 중: {index + 1}/{total_rows} ({val_f}) "
            f"- 새 파일: {generated_count}, 건너뜀: {skipped_count}"
        )

    st.success(f"사운드 파일 생성 완료! (새로 생성: {generated_count}, 건너뜀: {skipped_count})")
    return df


# ────────────────────────────────────────────
# Streamlit UI
# ────────────────────────────────────────────
st.set_page_config(page_title="영어 학습 자료 도구 세트", layout="wide")
st.title("🎧 영어 학습 자료 처리 도구")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🔤 단어 파일 처리", 
    "📚 리딩-해석추가", 
    "📑 리딩 스크립트 업로드", 
    "✍️ 리딩 단원 스크립트 추가",
    "🎧 리스닝 스크립트 업로드",
    "📊 리스닝 파일 처리",
    "📝 리스닝 단원 스크립트 추가"
])

# ==========================================
# TAB 1: 단어 파일 처리
# ==========================================
with tab1:
    st.markdown("""
    단어 엑셀 파일을 업로드하고, 실행할 기능을 **체크박스**로 선택한 뒤 **▶ 선택한 작업 실행** 버튼을 클릭하세요.  
    처리된 엑셀 파일은 원본 파일명에 `_update`가 붙어 저장됩니다.
    """)

    uploaded_voca = st.file_uploader("📁 단어 파일 업로더 (.xlsx)", type=["xlsx"], key="voca_uploader")

    if uploaded_voca is not None:
        voca_filename = uploaded_voca.name

        # 세션 상태에 단어 데이터프레임 저장
        if 'voca_df' not in st.session_state or 'voca_loaded_file' not in st.session_state \
                or st.session_state.voca_loaded_file != voca_filename:
            st.session_state.voca_df = pd.read_excel(uploaded_voca, header=0)
            st.session_state.voca_loaded_file = voca_filename
            st.info(f"✅ '{voca_filename}' 파일이 로드되었습니다.")

        v_df = st.session_state.voca_df

        st.write("### 📊 데이터 미리보기 (하단 5행)")
        st.dataframe(v_df.tail())

        base_output_dir = "output_sounds"

        # ── 체크박스 선택 영역 ──────────────────────
        st.write("---")
        st.write("### ⚙️ 실행할 작업 선택")

        col_left, col_right = st.columns(2)

        with col_left:
            do_typo    = st.checkbox("🔍 단어 오탈자 점검 (F열)", value=False, key="chk_typo")
            do_meaning  = st.checkbox("📖 의미 업데이트 (G열)", value=False, key="chk_meaning")
            do_ipa      = st.checkbox("🔤 발음기호 업데이트 (H열)", value=False, key="chk_ipa")

        with col_right:
            do_pos      = st.checkbox("🏷️ 품사 업데이트 (I열)", value=False, key="chk_pos")
            do_path     = st.checkbox("📂 사운드파일 경로 업데이트 (L열)", value=False, key="chk_path")
            do_sound    = st.checkbox("🎵 사운드 파일 생성 (MP3)", value=False, key="chk_sound")

        st.write("---")

        # ── 프로세스 버튼 ─────────────────────────
        if st.button("▶ 선택한 작업 실행", type="primary", use_container_width=True, key="btn_voca"):
            if not any([do_typo, do_meaning, do_ipa, do_pos, do_path, do_sound]):
                st.warning("⚠️ 실행할 작업을 하나 이상 선택해주세요.")
            else:
                with st.spinner("작업을 수행하는 중입니다..."):
                    current_v_df = st.session_state.voca_df.copy()

                    if do_typo:
                        st.write("#### 1️⃣ 단어 오탈자 점검")
                        current_v_df = check_typos(current_v_df)

                    if do_meaning:
                        st.write("#### 2️⃣ 의미 업데이트")
                        current_v_df = update_meaning(current_v_df)

                    if do_ipa:
                        st.write("#### 3️⃣ 발음기호 업데이트")
                        current_v_df = update_ipa(current_v_df, base_output_dir)

                    if do_pos:
                        st.write("#### 4️⃣ 품사 업데이트")
                        current_v_df = update_pos(current_v_df)

                    if do_path:
                        st.write("#### 5️⃣ 사운드파일 경로 업데이트")
                        current_v_df = update_sound_paths(current_v_df, base_output_dir)

                    if do_sound:
                        st.write("#### 6️⃣ 사운드 파일 생성")
                        current_v_df = generate_sounds(current_v_df, base_output_dir)

                    st.session_state.voca_df = current_v_df

                st.success("✅ 모든 선택 작업이 완료되었습니다!")

        # ── 다운로드 영역 ─────────────────────────
        st.write("---")
        st.write("### 📥 결과 엑셀 파일 다운로드")

        v_output = io.BytesIO()
        with pd.ExcelWriter(v_output, engine='openpyxl') as writer:
            st.session_state.voca_df.to_excel(writer, index=False)
        v_processed_data = v_output.getvalue()

        v_save_filename = get_update_filename(voca_filename)

        st.download_button(
            label=f"💾 '{v_save_filename}' 다운로드",
            data=v_processed_data,
            file_name=v_save_filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key="dl_voca"
        )

        st.info(f"🔊 사운드 파일 저장 위치: `{os.path.abspath(base_output_dir)}`")


# ==========================================
# TAB 2: 리딩 파일 처리
# ==========================================
with tab2:
    st.markdown("""
    리딩 엑셀 파일을 업로드하세요.  
    **D열**의 영어 문장을 읽어 **E열**이 비어있는 경우 한글로 해석하여 채워줍니다.
    """)

    # ── 템플릿 다운로드 기능 ──────────────────
    st.write("### 📥 리딩 파일 템플릿 다운로드")
    
    # 템플릿 데이터 생성
    template_main_cols = [
        "READING_BOOK_NM", "READING_UNIT_NM", "READING_SENTENCE_SEQ", 
        "READING_SENTENCE", "READING_SETENCE_MEANING", "READING_SENTENCE_SOUND_FILE"
    ]
    template_voca_cols = [
        "READING_BOOK_NM", "READING_UNIT_NM", 
        "READING_VOCA", "READING_VOCA_MEANING", "READING_VOCA_SOUND_FILE"
    ]
    template_links_cols = [
        "title", "link", "teacher_only"
    ]
    
    df_main_template = pd.DataFrame(columns=template_main_cols)
    df_voca_template = pd.DataFrame(columns=template_voca_cols)
    df_links_template = pd.DataFrame(columns=template_links_cols)
    
    template_buffer = io.BytesIO()
    with pd.ExcelWriter(template_buffer, engine='openpyxl') as writer:
        df_main_template.to_excel(writer, sheet_name='본문', index=False)
        df_voca_template.to_excel(writer, sheet_name='단어', index=False)
        df_links_template.to_excel(writer, sheet_name='links', index=False)
    
    st.download_button(
        label="📄 리딩 파일 템플릿(.xlsx) 다운로드",
        data=template_buffer.getvalue(),
        file_name="reading_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
        key="btn_template_download"
    )

    st.write("---")

    uploaded_reading = st.file_uploader("📁 리딩 파일 업로더 (.xlsx)", type=["xlsx"], key="reading_uploader")

    if uploaded_reading is not None:
        reading_filename = uploaded_reading.name

        # 세션 상태에 리딩 데이터프레임(딕셔너리) 저장
        if 'reading_sheets' not in st.session_state or 'reading_loaded_file' not in st.session_state \
                or st.session_state.reading_loaded_file != reading_filename:
            # 모든 시트 로드
            st.session_state.reading_sheets = pd.read_excel(uploaded_reading, sheet_name=None, header=0)
            st.session_state.reading_loaded_file = reading_filename
            st.info(f"✅ '{reading_filename}' 파일(모든 시트)이 로드되었습니다.")

        r_sheets = st.session_state.reading_sheets

        st.write("### 📊 데이터 미리보기 (시트 선택)")
        selected_sheet = st.selectbox("미리보기할 시트를 선택하세요:", list(r_sheets.keys()), key="sheet_selector")
        # 마지막 5개 컬럼만 미리보기에 표시
        st.dataframe(r_sheets[selected_sheet].tail(5))

        st.write("---")

        if st.button("▶ 모든 시트 해석/뜻 업데이트 실행", type="primary", use_container_width=True, key="btn_reading"):
            with st.spinner("번역 작업을 수행하는 중입니다..."):
                # 모든 시트 처리
                processed_sheets = process_reading_excel(st.session_state.reading_sheets.copy())
                st.session_state.reading_sheets = processed_sheets
            st.success("✅ 모든 시트의 업데이트가 완료되었습니다!")

        # ── 다운로드 영역 ─────────────────────────
        st.write("---")
        st.write("### 📥 결과 엑셀 파일 다운로드")

        r_output = io.BytesIO()
        with pd.ExcelWriter(r_output, engine='openpyxl') as writer:
            for sheet_name, sheet_df in st.session_state.reading_sheets.items():
                sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)
        r_processed_data = r_output.getvalue()

        r_save_filename = get_update_filename(reading_filename)

        st.download_button(
            label=f"💾 '{r_save_filename}' 다운로드",
            data=r_processed_data,
            file_name=r_save_filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key="dl_reading"
        )

# ==========================================
# TAB 3: 리딩 스크립트 업로드 (Word -> Excel)
# ==========================================
with tab3:
    st.markdown("""
    워드(.docx) 파일을 업로드하면 문장별로 분리하여 엑셀 형식으로 시트를 만들어줍니다.  
    - **Unit**으로 시작하는 문단은 단원명으로 인식합니다.
    - 문단이 바뀔 때 공백 행을 추가합니다.
    - 영어 문장을 한글로 자동 번역하여 E열에 추가합니다.
    """)

    uploaded_docx = st.file_uploader("📁 워드 파일 업로더 (.docx)", type=["docx"], key="docx_uploader")

    if uploaded_docx is not None:
        if st.button("▶ 엑셀 파일로 변환 실행", type="primary", use_container_width=True):
            try:
                from docx import Document
                from deep_translator import GoogleTranslator
                
                doc = Document(uploaded_docx)
                filename_base = Path(uploaded_docx.name).stem
                translator = GoogleTranslator(source='en', target='ko')
                
                # nltk sentence tokenizer 준비
                try:
                    nltk.data.find('tokenizers/punkt_tab')
                except LookupError:
                    nltk.download('punkt_tab', quiet=True)
                try:
                    nltk.data.find('tokenizers/punkt')
                except LookupError:
                    nltk.download('punkt', quiet=True)
                
                data = []
                current_unit = ""
                sentence_no = 1
                
                # 유효한 문단들만 필터링
                paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, p_text in enumerate(paragraphs):
                    # Unit 정보 업데이트 (예: 'Unit 01. The Sun')
                    if p_text.lower().startswith("unit"):
                        current_unit = p_text
                        sentence_no = 1  # 단원별로 문장 번호 초기화
                        # Unit 행은 데이터에 포함시키지 않거나, 필요에 따라 포함 가능
                        # 여기선 단원명으로만 사용하고 다음 문단부터 처리
                        continue
                    
                    # 문장 분리
                    sentences = nltk.sent_tokenize(p_text)
                    for sent in sentences:
                        # 한글 번역
                        try:
                            meaning = translator.translate(sent)
                        except:
                            meaning = ""
                        
                        # 사운드 경로 생성 규칙: Unit/파일명_번호.mp3
                        unit_folder = sanitize_filename(current_unit) if current_unit else "Unassigned"
                        sound_filename = f"{sanitize_filename(filename_base)}_{sentence_no}.mp3"
                        sound_path = f"{unit_folder}/{sound_filename}"
                        
                        data.append({
                            "A": filename_base,
                            "B": current_unit,
                            "C": sentence_no,
                            "D": sent,
                            "E": meaning,
                            "F": sound_path
                        })
                        sentence_no += 1
                    
                    # 문단이 바뀔 때 공백 행 추가 (사용자 요청: D열이 공백인 경우도 문장번호 기재)
                    if i < len(paragraphs) - 1:
                        # 다음 문단이 Unit으로 시작하면 공백행을 생략할 수도 있지만, 명시적 요청에 따라 일단 추가
                        data.append({
                            "A": filename_base,
                            "B": current_unit,
                            "C": sentence_no,
                            "D": "",
                            "E": "",
                            "F": ""
                        })
                        sentence_no += 1
                    
                    progress_bar.progress((i + 1) / len(paragraphs))
                    status_text.text(f"처리 중: {i + 1}/{len(paragraphs)} 문단 완료")
                
                result_df = pd.DataFrame(data)
                # 컬럼명 변경
                result_df.columns = [
                    "파일명", "Unit명", "문장번호", "영어문장", "한글해석", "사운드경로"
                ]
                
                st.success("✅ 변환 완료!")
                st.dataframe(result_df)
                
                # 다운로드 버튼
                docx_output = io.BytesIO()
                with pd.ExcelWriter(docx_output, engine='openpyxl') as writer:
                    result_df.to_excel(writer, index=False, sheet_name="본문")
                
                st.download_button(
                    label="📥 변환된 엑셀 파일 다운로드",
                    data=docx_output.getvalue(),
                    file_name=f"{filename_base}_converted.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    key="dl_docx"
                )
                
            except Exception as e:
                st.error(f"변환 중 오류 발생: {e}")
                st.exception(e)

# ==========================================
# TAB 4: 리딩 단원 스크립트 추가 (수동 입력)
# ==========================================
with tab4:
    st.markdown("""
    교재 제목, 단원 제목, 그리고 본문 스크립트를 직접 입력하여 엑셀 파일로 변환합니다.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        manual_book_nm = st.text_input("📚 교재 제목", placeholder="예: Middle School English 1")
    with col2:
        manual_unit_nm = st.text_input("📑 단원 제목", placeholder="예: Unit 1. Nice to Meet You")
        
    manual_body = st.text_area("📝 본문 입력 (여러 줄)", height=350, placeholder="여기에 본문 내용을 붙여넣으세요...")

    if st.button("▶ 입력 완료 및 엑셀 생성", type="primary", use_container_width=True):
        if not manual_book_nm or not manual_unit_nm or not manual_body:
            st.warning("⚠️ 교재 제목, 단원 제목, 본문을 모두 입력해주세요.")
        else:
            with st.spinner("처리 중입니다..."):
                try:
                    from deep_translator import GoogleTranslator
                    translator = GoogleTranslator(source='en', target='ko')
                    
                    # nltk 준비
                    try:
                        nltk.data.find('tokenizers/punkt_tab')
                    except LookupError:
                        nltk.download('punkt_tab', quiet=True)
                    
                    # 본문을 문단별로 나누고 문장으로 분리
                    paragraphs = [p.strip() for p in manual_body.split('\n') if p.strip()]
                    
                    manual_data = []
                    sentence_no = 1
                    
                    for i, p_text in enumerate(paragraphs):
                        sentences = nltk.sent_tokenize(p_text)
                        for sent in sentences:
                            try:
                                meaning = translator.translate(sent)
                            except:
                                meaning = ""
                            
                            manual_data.append({
                                "파일명": manual_book_nm,
                                "Unit명": manual_unit_nm,
                                "문장번호": sentence_no,
                                "영어문장": sent,
                                "한글해석": meaning,
                                "사운드경로": f"{sanitize_filename(manual_unit_nm)}/{sanitize_filename(manual_book_nm)}_{sentence_no}.mp3"
                            })
                            sentence_no += 1
                        
                        # 문단 간 공백 행 추가
                        if i < len(paragraphs) - 1:
                            manual_data.append({
                                "파일명": manual_book_nm,
                                "Unit명": manual_unit_nm,
                                "문장번호": sentence_no,
                                "영어문장": "",
                                "한글해석": "",
                                "사운드경로": ""
                            })
                            sentence_no += 1
                            
                    manual_df = pd.DataFrame(manual_data)
                    st.success("✅ 변환이 완료되었습니다!")
                    st.dataframe(manual_df)
                    
                    # 다운로드
                    m_output = io.BytesIO()
                    with pd.ExcelWriter(m_output, engine='openpyxl') as writer:
                        manual_df.to_excel(writer, index=False, sheet_name="본문")
                    
                    st.download_button(
                        label="📥 생성된 엑셀 파일 다운로드",
                        data=m_output.getvalue(),
                        file_name=f"{manual_unit_nm}_script.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                        key="dl_manual"
                    )
                except Exception as e:
                    st.error(f"오류 발생: {e}")


# ==========================================
# TAB 5: 리스닝 스크립트 업로드
# ==========================================
with tab5:
    st.markdown("""
    워드(.docx) 파일을 업로드하면 대화 형식 및 문장을 분리하여 리스닝 엑셀 형식으로 시트를 만들어줍니다.  
    - **Unit**으로 시작하는 문단은 단원명으로 인식합니다.
    - 대화 형식(`Sally: Hello. How are you?`)은 화자 이름과 함께 문장별로 쪼개어 처리합니다.
    - 한 명이 여러 문장을 말한 경우 각각의 문장이 분리됩니다.
    - 문단이 바뀔 때는 D 열에 공백 행을 추가합니다. (C 열 문장번호는 유지)
    - 영어 문장을 한글로 자동 번역하여 E 열에 추가합니다.
    """)

    uploaded_listen_docx = st.file_uploader("📁 리스닝 워드 파일 업로더 (.docx)", type=["docx"], key="listen_docx_uploader")

    if uploaded_listen_docx is not None:
        if st.button("▶ 리스닝 엑셀 파일로 변환 실행", type="primary", use_container_width=True, key="btn_listen_docx_convert"):
            try:
                from docx import Document
                
                doc = Document(uploaded_listen_docx)
                filename_base = Path(uploaded_listen_docx.name).stem
                
                paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
                
                with st.spinner("리스닝 워드 파일을 분석하고 번역하는 중..."):
                    data = parse_listening_paragraphs(paragraphs, filename_base, "")
                
                if data:
                    result_df = pd.DataFrame(data)
                    result_df.columns = ["파일명", "Unit명", "문장번호", "영어문장", "한글해석", "사운드경로"]
                    
                    st.success("✅ 변환 완료!")
                    st.dataframe(result_df)
                    
                    # 다운로드 버튼
                    docx_output = io.BytesIO()
                    with pd.ExcelWriter(docx_output, engine='openpyxl') as writer:
                        result_df.to_excel(writer, index=False, sheet_name="본문")
                    
                    st.download_button(
                        label="📥 변환된 리스닝 엑셀 파일 다운로드",
                        data=docx_output.getvalue(),
                        file_name=f"{filename_base}_listening_converted.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                        key="dl_listen_docx"
                    )
                else:
                    st.warning("⚠️ 추출된 텍스트 데이터가 없습니다.")
                    
            except Exception as e:
                st.error(f"변환 중 오류 발생: {e}")
                st.exception(e)


# ==========================================
# TAB 6: 리스닝 파일 처리
# ==========================================
with tab6:
    st.markdown("""
    리스닝 엑셀 파일을 업로드하세요.  
    **D열**의 영어 문장을 읽어 **E열**이 비어있는 경우 한글로 해석하여 채워줍니다.  
    단, `"Sally:"`와 같이 콜론으로 끝나는 화자 표시 텍스트는 번역하지 않고 그대로 유지합니다.
    """)

    # ── 템플릿 다운로드 기능 ──────────────────
    st.write("### 📥 리스닝 파일 템플릿 다운로드")
    
    template_listen_main_cols = [
        "LISTENING_BOOK_NM", "LISTENING_UNIT_NM", "LISTENING_SENTENCE_SEQ", 
        "LISTENING_SENTENCE", "LISTENING_SETENCE_MEANING", "LISTENING_SENTENCE_SOUND_FILE"
    ]
    template_listen_voca_cols = [
        "LISTENING_BOOK_NM", "LISTENING_UNIT_NM", 
        "LISTENING_VOCA", "LISTENING_VOCA_MEANING", "LISTENING_VOCA_SOUND_FILE"
    ]
    template_listen_links_cols = [
        "title", "link", "teacher_only"
    ]
    
    df_listen_main_template = pd.DataFrame(columns=template_listen_main_cols)
    df_listen_voca_template = pd.DataFrame(columns=template_listen_voca_cols)
    df_listen_links_template = pd.DataFrame(columns=template_listen_links_cols)
    
    listen_template_buffer = io.BytesIO()
    with pd.ExcelWriter(listen_template_buffer, engine='openpyxl') as writer:
        df_listen_main_template.to_excel(writer, sheet_name='본문', index=False)
        df_listen_voca_template.to_excel(writer, sheet_name='단어', index=False)
        df_listen_links_template.to_excel(writer, sheet_name='links', index=False)
    
    st.download_button(
        label="📄 리스닝 파일 템플릿(.xlsx) 다운로드",
        data=listen_template_buffer.getvalue(),
        file_name="listening_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
        key="btn_listen_template_download"
    )

    st.write("---")

    uploaded_listening = st.file_uploader("📁 리스닝 파일 업로더 (.xlsx)", type=["xlsx"], key="listening_uploader")

    if uploaded_listening is not None:
        listening_filename = uploaded_listening.name

        # 세션 상태에 리스닝 데이터프레임(딕셔너리) 저장
        if 'listening_sheets' not in st.session_state or 'listening_loaded_file' not in st.session_state \
                or st.session_state.listening_loaded_file != listening_filename:
            st.session_state.listening_sheets = pd.read_excel(uploaded_listening, sheet_name=None, header=0)
            st.session_state.listening_loaded_file = listening_filename
            st.info(f"✅ '{listening_filename}' 파일(모든 시트)이 로드되었습니다.")

        l_sheets = st.session_state.listening_sheets

        st.write("### 📊 데이터 미리보기 (시트 선택)")
        selected_l_sheet = st.selectbox("미리보기할 시트를 선택하세요:", list(l_sheets.keys()), key="l_sheet_selector")
        st.dataframe(l_sheets[selected_l_sheet].tail(5))

        st.write("---")

        if st.button("▶ 모든 시트 해석/뜻 업데이트 실행", type="primary", use_container_width=True, key="btn_listening_process"):
            with st.spinner("번역 작업을 수행하는 중입니다..."):
                processed_l_sheets = process_listening_excel(st.session_state.listening_sheets.copy())
                st.session_state.listening_sheets = processed_l_sheets
            st.success("✅ 모든 시트의 업데이트가 완료되었습니다!")

        # ── 다운로드 영역 ─────────────────────────
        st.write("---")
        st.write("### 📥 결과 엑셀 파일 다운로드")

        l_output = io.BytesIO()
        with pd.ExcelWriter(l_output, engine='openpyxl') as writer:
            for sheet_name, sheet_df in st.session_state.listening_sheets.items():
                sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)
        l_processed_data = l_output.getvalue()

        l_save_filename = get_update_filename(listening_filename)

        st.download_button(
            label=f"💾 '{l_save_filename}' 다운로드",
            data=l_processed_data,
            file_name=l_save_filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key="dl_listening_processed"
        )


# ==========================================
# TAB 7: 리스닝 단원 스크립트 추가 (수동 입력)
# ==========================================
with tab7:
    st.markdown("""
    교재 제목, 단원 제목, 화자 목록, 그리고 대화 스크립트를 직접 입력하여 리스닝용 엑셀 파일로 변환합니다.
    """)
    
    col1_l, col2_l = st.columns(2)
    with col1_l:
        manual_listen_book_nm = st.text_input("📚 교재 제목", placeholder="예: Middle School Listening 1", key="input_l_book_nm")
    with col2_l:
        manual_listen_unit_nm = st.text_input("📑 단원 제목", placeholder="예: Unit 1. Nice to Meet You", key="input_l_unit_nm")
        
    manual_listen_speakers = st.text_input("👤 화자 목록 (쉼표로 구분)", placeholder="예: Sally, John, Teacher", key="input_l_speakers")
    manual_listen_body = st.text_area("📝 본문 입력 (여러 줄)", height=300, placeholder="여기에 대화 내용을 붙여넣으세요...", key="input_l_body")

    # 실시간 미리보기 기능 추가
    if manual_listen_body:
        st.write("---")
        st.write("### 🔍 입력 본문 미리보기 (화자 강조)")
        st.info("💡 화자 이름이 빨간색 볼드체로 표시됩니다. 줄바꿈(문단 구분) 상태를 확인 후 하단의 엑셀 생성 버튼을 눌러주세요.")
        
        # 화자 목록 파싱
        speakers = [s.strip() for s in manual_listen_speakers.split(",") if s.strip()] if manual_listen_speakers else []
        
        # HTML 렌더링을 통한 화자 강조 및 개행 유지
        preview_html = manual_listen_body.replace("\n", "<br>")
        if speakers:
            for sp in speakers:
                # 단어 경계(\b)를 고려한 정규식으로 화자 강조 (예외 문자가 들어갈 수 있어 re.escape 사용)
                pattern = re.compile(rf"\b({re.escape(sp)})\b", re.IGNORECASE)
                preview_html = pattern.sub(r'<span style="color:red; font-weight:bold;">\1</span>', preview_html)
        
        st.markdown(
            f'<div style="border:1px solid #555; padding:15px; border-radius:5px; line-height:1.6;">{preview_html}</div>', 
            unsafe_allow_html=True
        )
        st.write("---")

    if st.button("▶ 리스닝 엑셀 생성", type="primary", use_container_width=True, key="btn_listen_manual_create"):
        if not manual_listen_book_nm or not manual_listen_unit_nm or not manual_listen_body:
            st.warning("⚠️ 교재 제목, 단원 제목, 본문을 모두 입력해주세요.")
        else:
            with st.spinner("처리 중입니다..."):
                try:
                    paragraphs = [p.strip() for p in manual_listen_body.split('\n') if p.strip()]
                    
                    # 화자 목록 파싱
                    speakers = [s.strip() for s in manual_listen_speakers.split(",") if s.strip()] if manual_listen_speakers else []
                    
                    manual_listen_data = parse_listening_paragraphs(paragraphs, manual_listen_book_nm, manual_listen_unit_nm, speakers=speakers)
                    
                    if manual_listen_data:
                        manual_listen_df = pd.DataFrame(manual_listen_data)
                        manual_listen_df.columns = ["파일명", "Unit명", "문장번호", "영어문장", "한글해석", "사운드경로"]
                        
                        st.success("✅ 변환이 완료되었습니다!")
                        st.dataframe(manual_listen_df)
                        
                        # 다운로드
                        ml_output = io.BytesIO()
                        with pd.ExcelWriter(ml_output, engine='openpyxl') as writer:
                            manual_listen_df.to_excel(writer, index=False, sheet_name="본문")
                        
                        st.download_button(
                            label="📥 생성된 리스닝 엑셀 파일 다운로드",
                            data=ml_output.getvalue(),
                            file_name=f"{manual_listen_unit_nm}_listening_script.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True,
                            key="dl_listen_manual"
                        )
                    else:
                        st.warning("⚠️ 처리할 텍스트 데이터가 생성되지 않았습니다.")
                except Exception as e:
                    st.error(f"오류 발생: {e}")
