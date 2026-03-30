"""
eval_code.py — Code Based Evaluation (Streamlit)

[평가 지표]
  1. 유사도 점수 : bge-m3-korean 임베딩 → 코사인 유사도 (응답 vs 수기 정답지)
  2. 카드명 정확도 : 정답 카드 중 응답에 포함된 비율 (Recall)

[탭 구성]
  Tab 1. 정답지 관리  — 페르소나별 참조 텍스트 + 정답 카드명 수기 작성/저장
  Tab 2. 유사도 평가  — 임베딩 계산 + 카드명 매칭 실행
  Tab 3. 결과 분석   — 모델 비교 통계 + 페르소나별 차트
"""

import os
import json
import glob
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from sentence_transformers import SentenceTransformer

# ==========================================
# 전역 설정
# ==========================================
EVAL_DIR = "eval_results/최종 결과/Model Based/3.5Turbo_vs_4o"
GROUND_TRUTH_FILE = os.path.join(EVAL_DIR, "ground_truth.json")

CODE_EVAL_DIRS = {
    "3.5Turbo_vs_4o": "eval_results/최종 결과/Model Based/3.5Turbo_vs_4o/Code_eval",
    "4o_vs_4o":       "eval_results/최종 결과/Model Based/4o_vs_4o/Code_eval",
}


# ==========================================
# 리소스 및 유틸
# ==========================================
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("upskyy/bge-m3-korean")


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def card_recall(predicted_cards: list[str], expected_cards: list[str]) -> float:
    """정답 카드 중 추천 카드에 포함된 비율 (부분 문자열 매칭)"""
    if not expected_cards:
        return 0.0
    hits = sum(
        1
        for exp in expected_cards
        if any(exp in pred or pred in exp for pred in predicted_cards)
    )
    return hits / len(expected_cards)


def load_ground_truth() -> dict:
    if os.path.exists(GROUND_TRUTH_FILE):
        with open(GROUND_TRUTH_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_ground_truth(data: dict):
    os.makedirs(EVAL_DIR, exist_ok=True)
    with open(GROUND_TRUTH_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_eval_files() -> list[str]:
    files = sorted(glob.glob(os.path.join(EVAL_DIR, "*.json")))
    return [f for f in files if "ground_truth" not in f and "_corrected" not in f]


# ==========================================
# 메인
# ==========================================
def main():
    st.set_page_config(page_title="Code Based Evaluation", layout="wide")
    st.title("Code Based Evaluation — bge-m3-korean 유사도")

    eval_files = get_eval_files()
    ground_truth = load_ground_truth()

    tab1, tab2, tab3, tab4 = st.tabs(["📝 정답지 관리", "🔬 평가 실행", "📊 결과 분석", "📈 병합 통계"])

    # ──────────────────────────────────────────────────────────────────
    # TAB 1: 정답지 관리
    # ──────────────────────────────────────────────────────────────────
    with tab1:
        st.header("페르소나별 정답지 작성")
        st.caption("카드고릴라에서 직접 확인한 최적 카드 정보를 입력하세요.")

        if not eval_files:
            st.warning(f"`{EVAL_DIR}/` 폴더에 평가 파일이 없습니다.")
        else:
            ref_file = st.selectbox(
                "페르소나 목록 기준 파일",
                eval_files,
                format_func=os.path.basename,
                key="gt_ref_file",
            )
            with open(ref_file, "r", encoding="utf-8") as f:
                ref_data = json.load(f)
            personas = [p["persona"] for p in ref_data.get("results", [])]

            selected_persona = st.selectbox("편집할 페르소나", personas)
            current = ground_truth.get(selected_persona, {})

            st.markdown("#### 참조 텍스트 (모범 답안)")
            st.caption(
                "이 페르소나에게 추천해야 할 카드명, 핵심 혜택, 추천 이유를 구체적으로 작성하세요. "
                "유사도 점수의 기준이 되므로 혜택 수치·조건을 포함할수록 변별력이 높아집니다."
            )
            ref_text = st.text_area(
                "참조 텍스트",
                value=current.get("reference_text", ""),
                height=220,
                key=f"ref_{selected_persona}",
                placeholder=(
                    "예) 노리2 체크카드 — 배달앱(배달의민족·요기요) 월 1,000원 할인(전월실적 20만원), "
                    "스타벅스·커피빈 월 최대 3,000원 할인. 월 15만원 배달·12만원 카페 지출 패턴에 최적."
                ),
            )

            st.markdown("#### 정답 카드명 (쉼표 구분)")
            st.caption(
                "응답에 반드시 포함되어야 할 카드명을 입력하세요. 카드명 Recall 계산에 사용됩니다."
            )
            card_input = st.text_input(
                "정답 카드명",
                value=", ".join(current.get("expected_cards", [])),
                key=f"cards_{selected_persona}",
                placeholder="예: 노리2 체크카드(KB Pay), 현대카드Z everyday",
            )

            if st.button("💾 저장", type="primary", key=f"save_{selected_persona}"):
                expected_cards = [c.strip() for c in card_input.split(",") if c.strip()]
                ground_truth[selected_persona] = {
                    "reference_text": ref_text.strip(),
                    "expected_cards": expected_cards,
                }
                save_ground_truth(ground_truth)
                st.success(f"**{selected_persona}** 정답지 저장 완료")
                st.rerun()

            # 전체 현황 테이블
            st.divider()
            st.markdown("#### 정답지 작성 현황")
            rows = []
            for p in personas:
                gt = ground_truth.get(p, {})
                rows.append(
                    {
                        "페르소나": p,
                        "참조 텍스트": "✅" if gt.get("reference_text") else "❌",
                        "정답 카드 수": len(gt.get("expected_cards", [])),
                        "정답 카드": ", ".join(gt.get("expected_cards", [])) or "—",
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # ──────────────────────────────────────────────────────────────────
    # TAB 2: 평가 실행
    # ──────────────────────────────────────────────────────────────────
    with tab2:
        st.header("유사도 평가 실행")

        if not eval_files:
            st.warning(f"`{EVAL_DIR}/` 폴더에 평가 파일이 없습니다.")
        else:
            selected_eval = st.selectbox(
                "평가 파일 선택",
                eval_files,
                format_func=os.path.basename,
                key="run_eval_file",
            )

            with open(selected_eval, "r", encoding="utf-8") as f:
                eval_data = json.load(f)

            key_x = eval_data.get("key_x", "model_x")
            key_y = eval_data.get("key_y", "model_y")
            st.caption(f"비교 모델: `{key_x}` (X) vs `{key_y}` (Y)")

            personas_list = eval_data.get("results", [])
            gt_ready = sum(
                1
                for p in personas_list
                if ground_truth.get(p["persona"], {}).get("reference_text")
            )
            total_p = len(personas_list)

            col_info, col_btn = st.columns([3, 1])
            col_info.info(
                f"정답지 완성: **{gt_ready} / {total_p}** 페르소나  (미완성 페르소나는 스킵)"
            )

            run_btn = col_btn.button(
                "평가 시작",
                type="primary",
                use_container_width=True,
                disabled=(gt_ready == 0),
            )

            if run_btn:
                model = load_embedding_model()
                progress = st.progress(0.0, text="임베딩 계산 중...")
                rows = []

                for idx, persona_data in enumerate(personas_list):
                    persona = persona_data["persona"]
                    gt = ground_truth.get(persona, {})
                    ref_text = gt.get("reference_text", "")
                    expected_cards = gt.get("expected_cards", [])

                    progress.progress(
                        (idx + 1) / total_p,
                        text=f"[{idx+1}/{total_p}] {persona}",
                    )

                    if not ref_text:
                        continue

                    ref_emb = model.encode(ref_text, normalize_embeddings=True)

                    for run_data in persona_data.get("runs", []):
                        resp_x = run_data.get("resp_x", "")
                        resp_y = run_data.get("resp_y", "")
                        cards_x = run_data.get("cards_x", [])
                        cards_y = run_data.get("cards_y", [])

                        if not resp_x or not resp_y:
                            continue

                        emb_x = model.encode(resp_x, normalize_embeddings=True)
                        emb_y = model.encode(resp_y, normalize_embeddings=True)

                        rows.append(
                            {
                                "persona": persona,
                                "run": run_data.get("run", 0),
                                "sim_x": cosine_sim(ref_emb, emb_x),
                                "sim_y": cosine_sim(ref_emb, emb_y),
                                "recall_x": card_recall(cards_x, expected_cards),
                                "recall_y": card_recall(cards_y, expected_cards),
                            }
                        )

                progress.progress(1.0, text="완료!")

                if rows:
                    st.session_state["code_df"] = pd.DataFrame(rows)
                    st.session_state["code_kx"] = key_x
                    st.session_state["code_ky"] = key_y
                    st.success(f"평가 완료 — {len(rows)}개 Run 분석됨")
                else:
                    st.warning("유효한 결과가 없습니다. 정답지를 먼저 작성하세요.")

    # ──────────────────────────────────────────────────────────────────
    # TAB 3: 결과 분석
    # ──────────────────────────────────────────────────────────────────
    with tab3:
        st.header("결과 분석")

        if "code_df" not in st.session_state:
            st.info("먼저 [평가 실행] 탭에서 평가를 실행하세요.")
            return

        df: pd.DataFrame = st.session_state["code_df"]
        kx: str = st.session_state["code_kx"]
        ky: str = st.session_state["code_ky"]

        # ── 전체 통계 카드 ─────────────────────────────────────────────
        st.subheader("전체 통계")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            f"{kx} 평균 유사도",
            f"{df['sim_x'].mean():.3f}",
            f"σ = {df['sim_x'].std():.3f}",
        )
        c2.metric(
            f"{ky} 평균 유사도",
            f"{df['sim_y'].mean():.3f}",
            f"σ = {df['sim_y'].std():.3f}",
        )
        c3.metric(f"{kx} 카드 Recall", f"{df['recall_x'].mean()*100:.1f}%")
        c4.metric(f"{ky} 카드 Recall", f"{df['recall_y'].mean()*100:.1f}%")

        # 유사도 기준 승률
        df["sim_winner"] = df.apply(
            lambda r: (
                kx
                if r["sim_x"] > r["sim_y"]
                else (ky if r["sim_y"] > r["sim_x"] else "tie")
            ),
            axis=1,
        )
        n = len(df)
        wx = (df["sim_winner"] == kx).sum()
        wy = (df["sim_winner"] == ky).sum()
        wt = (df["sim_winner"] == "tie").sum()
        st.caption(
            f"유사도 기준 승률 — **{kx}**: {wx/n*100:.1f}%  |  "
            f"**{ky}**: {wy/n*100:.1f}%  |  Tie: {wt}"
        )

        st.divider()

        # ── 페르소나별 평균 유사도 차트 ─────────────────────────────────
        st.subheader("페르소나별 평균 유사도")
        sim_rows = []
        for persona, grp in df.groupby("persona"):
            sim_rows += [
                {
                    "페르소나": persona,
                    "모델": kx,
                    "평균 유사도": grp["sim_x"].mean(),
                    "std": grp["sim_x"].std(),
                },
                {
                    "페르소나": persona,
                    "모델": ky,
                    "평균 유사도": grp["sim_y"].mean(),
                    "std": grp["sim_y"].std(),
                },
            ]
        sim_df = pd.DataFrame(sim_rows).fillna(0)

        sim_chart = (
            alt.Chart(sim_df)
            .mark_bar(opacity=0.85)
            .encode(
                x=alt.X("평균 유사도:Q", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("페르소나:N"),
                color=alt.Color("모델:N"),
                xOffset="모델:N",
                tooltip=[
                    "페르소나",
                    "모델",
                    alt.Tooltip("평균 유사도:Q", format=".3f"),
                    alt.Tooltip("std:Q", format=".3f", title="σ"),
                ],
            )
            .properties(height=300)
        )
        st.altair_chart(sim_chart, use_container_width=True)

        # ── 페르소나별 카드 Recall 차트 ─────────────────────────────────
        st.subheader("페르소나별 카드명 Recall (정답 카드 포함률)")
        recall_rows = []
        for persona, grp in df.groupby("persona"):
            recall_rows += [
                {
                    "페르소나": persona,
                    "모델": kx,
                    "카드 Recall": grp["recall_x"].mean(),
                },
                {
                    "페르소나": persona,
                    "모델": ky,
                    "카드 Recall": grp["recall_y"].mean(),
                },
            ]
        rc_df = pd.DataFrame(recall_rows)

        rc_chart = (
            alt.Chart(rc_df)
            .mark_bar(opacity=0.85)
            .encode(
                x=alt.X(
                    "카드 Recall:Q",
                    scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(format=".0%"),
                ),
                y=alt.Y("페르소나:N"),
                color=alt.Color("모델:N"),
                xOffset="모델:N",
                tooltip=[
                    "페르소나",
                    "모델",
                    alt.Tooltip("카드 Recall:Q", format=".1%"),
                ],
            )
            .properties(height=300)
        )
        st.altair_chart(rc_chart, use_container_width=True)

        # ── 종합 산점도: 유사도 vs 카드 Recall ──────────────────────────
        st.subheader("유사도 vs 카드 Recall 산점도")
        scatter_rows = pd.concat(
            [
                df[["persona", "run", "sim_x", "recall_x"]]
                .rename(columns={"sim_x": "유사도", "recall_x": "카드 Recall"})
                .assign(모델=kx),
                df[["persona", "run", "sim_y", "recall_y"]]
                .rename(columns={"sim_y": "유사도", "recall_y": "카드 Recall"})
                .assign(모델=ky),
            ],
            ignore_index=True,
        )
        scatter = (
            alt.Chart(scatter_rows)
            .mark_circle(size=60, opacity=0.6)
            .encode(
                x=alt.X("유사도:Q", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y(
                    "카드 Recall:Q",
                    scale=alt.Scale(domain=[-0.05, 1.05]),
                    axis=alt.Axis(format=".0%"),
                ),
                color=alt.Color("모델:N"),
                tooltip=[
                    "persona",
                    "run",
                    "모델",
                    alt.Tooltip("유사도:Q", format=".3f"),
                    alt.Tooltip("카드 Recall:Q", format=".0%"),
                ],
            )
            .properties(height=300)
        )
        st.altair_chart(scatter, use_container_width=True)

        # ── 상세 테이블 ─────────────────────────────────────────────────
        with st.expander("Run별 상세 결과"):
            display = df.rename(
                columns={
                    "sim_x": f"{kx} 유사도",
                    "sim_y": f"{ky} 유사도",
                    "recall_x": f"{kx} 카드Recall",
                    "recall_y": f"{ky} 카드Recall",
                    "sim_winner": "유사도 승자",
                }
            )
            st.dataframe(display, use_container_width=True)

        # ── 결과 저장 ────────────────────────────────────────────────────
        if st.button("📁 결과 JSON 저장"):
            out_path = os.path.join(EVAL_DIR, f"code_eval_{kx}_vs_{ky}.json")
            df.to_json(out_path, orient="records", force_ascii=False, indent=2)
            st.success(f"저장 완료: `{out_path}`")

    # ──────────────────────────────────────────────────────────────────
    # TAB 4: 병합 통계
    # ──────────────────────────────────────────────────────────────────
    with tab4:
        st.header("최종 결과 병합 통계")
        st.caption("Code_eval 디렉토리의 JSON 파일을 선택해 병합하고 통계를 확인합니다.")

        # ── 디렉토리별 파일 선택 ──────────────────────────────────────
        selected_per_dir: dict[str, list[str]] = {}
        for dir_label, dirpath in CODE_EVAL_DIRS.items():
            files = sorted(glob.glob(os.path.join(dirpath, "*.json"))) if os.path.isdir(dirpath) else []
            if not files:
                continue
            with st.expander(f"📂 {dir_label}  ({len(files)}개 파일)", expanded=True):
                chosen = st.multiselect(
                    "병합할 파일",
                    files,
                    default=files,
                    format_func=os.path.basename,
                    key=f"merge4_sel_{dir_label}",
                )
                if chosen:
                    selected_per_dir[dir_label] = chosen

        if not selected_per_dir:
            st.info("Code_eval 파일이 없습니다. 디렉토리를 확인하세요.")
        elif st.button("📊 병합 및 통계 계산", type="primary", key="merge4_btn"):
            merged: dict[str, pd.DataFrame] = {}
            for label, file_list in selected_per_dir.items():
                records = []
                kx_m, ky_m = "model_c", "model_b"  # 파일명 기본값
                for fp in file_list:
                    # 파일명에서 모델 키 추출: code_eval_{kx}_vs_{ky}.json
                    base = os.path.basename(fp)
                    import re
                    m = re.search(r"code_eval_(.+?)_vs_(.+?)\.json", base)
                    if m:
                        kx_m, ky_m = m.group(1), m.group(2)
                    with open(fp, "r", encoding="utf-8") as f:
                        records.extend(json.load(f))
                if records:
                    mdf = pd.DataFrame(records)
                    mdf["_kx"] = kx_m
                    mdf["_ky"] = ky_m
                    merged[label] = mdf
            st.session_state["merge4_data"] = merged

        if "merge4_data" in st.session_state and st.session_state["merge4_data"]:
            merged = st.session_state["merge4_data"]

            # ── 요약 통계 카드 ────────────────────────────────────────
            st.divider()
            st.subheader("요약 통계")
            sum_cols = st.columns(len(merged))
            for col, (label, mdf) in zip(sum_cols, merged.items()):
                kx_m = mdf["_kx"].iloc[0]
                ky_m = mdf["_ky"].iloc[0]
                n = len(mdf)
                wx = (mdf["sim_winner"] == kx_m).sum() if "sim_winner" in mdf.columns else 0
                wy = (mdf["sim_winner"] == ky_m).sum() if "sim_winner" in mdf.columns else 0
                with col:
                    st.markdown(f"**{label}** ({n}건)")
                    c1, c2 = st.columns(2)
                    c1.metric(f"{kx_m} 평균 유사도", f"{mdf['sim_x'].mean():.3f}", f"σ={mdf['sim_x'].std():.3f}")
                    c2.metric(f"{ky_m} 평균 유사도", f"{mdf['sim_y'].mean():.3f}", f"σ={mdf['sim_y'].std():.3f}")
                    c1.metric(f"{kx_m} 카드 Recall", f"{mdf['recall_x'].mean()*100:.1f}%")
                    c2.metric(f"{ky_m} 카드 Recall", f"{mdf['recall_y'].mean()*100:.1f}%")
                    st.caption(f"유사도 승률 — {kx_m}: {wx/n*100:.1f}%  |  {ky_m}: {wy/n*100:.1f}%")

            # ── 데이터셋 간 비교 차트 ─────────────────────────────────
            if len(merged) > 1:
                st.divider()
                st.subheader("데이터셋 간 평균 유사도 비교")
                cmp_rows = []
                for label, mdf in merged.items():
                    kx_m, ky_m = mdf["_kx"].iloc[0], mdf["_ky"].iloc[0]
                    cmp_rows += [
                        {"데이터셋": label, "모델": kx_m, "평균 유사도": round(mdf["sim_x"].mean(), 4)},
                        {"데이터셋": label, "모델": ky_m, "평균 유사도": round(mdf["sim_y"].mean(), 4)},
                    ]
                cmp_df = pd.DataFrame(cmp_rows)
                st.altair_chart(
                    alt.Chart(cmp_df).mark_bar().encode(
                        x=alt.X("데이터셋:N"),
                        y=alt.Y("평균 유사도:Q", scale=alt.Scale(domain=[0, 1])),
                        color=alt.Color("모델:N"),
                        xOffset="모델:N",
                        tooltip=["데이터셋", "모델", alt.Tooltip("평균 유사도:Q", format=".4f")],
                    ).properties(height=280),
                    use_container_width=True,
                )

                st.subheader("데이터셋 간 카드 Recall 비교")
                rc_rows = []
                for label, mdf in merged.items():
                    kx_m, ky_m = mdf["_kx"].iloc[0], mdf["_ky"].iloc[0]
                    rc_rows += [
                        {"데이터셋": label, "모델": kx_m, "카드 Recall": round(mdf["recall_x"].mean(), 4)},
                        {"데이터셋": label, "모델": ky_m, "카드 Recall": round(mdf["recall_y"].mean(), 4)},
                    ]
                rc_df = pd.DataFrame(rc_rows)
                st.altair_chart(
                    alt.Chart(rc_df).mark_bar().encode(
                        x=alt.X("데이터셋:N"),
                        y=alt.Y("카드 Recall:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format=".0%")),
                        color=alt.Color("모델:N"),
                        xOffset="모델:N",
                        tooltip=["데이터셋", "모델", alt.Tooltip("카드 Recall:Q", format=".1%")],
                    ).properties(height=280),
                    use_container_width=True,
                )

            # ── 데이터셋별 페르소나 차트 ──────────────────────────────
            st.divider()
            st.subheader("페르소나별 상세 차트")
            for label, mdf in merged.items():
                kx_m, ky_m = mdf["_kx"].iloc[0], mdf["_ky"].iloc[0]
                with st.expander(f"📈 {label}", expanded=False):
                    sim_rows = []
                    for persona, grp in mdf.groupby("persona"):
                        sim_rows += [
                            {"페르소나": persona, "모델": kx_m, "평균 유사도": grp["sim_x"].mean()},
                            {"페르소나": persona, "모델": ky_m, "평균 유사도": grp["sim_y"].mean()},
                        ]
                    st.altair_chart(
                        alt.Chart(pd.DataFrame(sim_rows)).mark_bar(opacity=0.85).encode(
                            x=alt.X("평균 유사도:Q", scale=alt.Scale(domain=[0, 1])),
                            y=alt.Y("페르소나:N"),
                            color=alt.Color("모델:N"),
                            xOffset="모델:N",
                            tooltip=["페르소나", "모델", alt.Tooltip("평균 유사도:Q", format=".3f")],
                        ).properties(height=300),
                        use_container_width=True,
                    )

            # ── JSON 내보내기 ─────────────────────────────────────────
            st.divider()
            export: dict = {}
            for label, mdf in merged.items():
                kx_m, ky_m = mdf["_kx"].iloc[0], mdf["_ky"].iloc[0]
                n = len(mdf)
                wx = int((mdf["sim_winner"] == kx_m).sum()) if "sim_winner" in mdf.columns else 0
                wy = int((mdf["sim_winner"] == ky_m).sum()) if "sim_winner" in mdf.columns else 0
                persona_details = []
                for persona, grp in mdf.groupby("persona"):
                    persona_details.append({
                        "persona": persona,
                        "total_runs": len(grp),
                        f"{kx_m}_avg_sim": round(grp["sim_x"].mean(), 4),
                        f"{ky_m}_avg_sim": round(grp["sim_y"].mean(), 4),
                        f"{kx_m}_avg_recall": round(grp["recall_x"].mean(), 4),
                        f"{ky_m}_avg_recall": round(grp["recall_y"].mean(), 4),
                    })
                export[label] = {
                    "dataset": label,
                    "model_x": kx_m,
                    "model_y": ky_m,
                    "total_runs": n,
                    "summary": {
                        f"{kx_m}_avg_sim": round(mdf["sim_x"].mean(), 4),
                        f"{ky_m}_avg_sim": round(mdf["sim_y"].mean(), 4),
                        f"{kx_m}_avg_recall": round(mdf["recall_x"].mean(), 4),
                        f"{ky_m}_avg_recall": round(mdf["recall_y"].mean(), 4),
                        f"{kx_m}_sim_winrate_pct": round(wx / n * 100, 1) if n else 0,
                        f"{ky_m}_sim_winrate_pct": round(wy / n * 100, 1) if n else 0,
                    },
                    "persona_details": persona_details,
                }
            export_json = json.dumps(export, ensure_ascii=False, indent=2)
            st.download_button(
                label="⬇️ 병합 통계 JSON 다운로드",
                data=export_json.encode("utf-8"),
                file_name="merged_code_eval_stats.json",
                mime="application/json",
                use_container_width=True,
                type="primary",
            )
            with st.expander("미리보기"):
                st.json(export)


if __name__ == "__main__":
    main()
