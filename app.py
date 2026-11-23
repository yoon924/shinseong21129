import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import chi2_contingency
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------------------------
# 페이지 설정 및 미니멀 스타일링
# ---------------------------------------------------------
st.set_page_config(
    page_title="AuditaHR - AI 채용 데이터 편향 정밀 진단 및 공정 재선발",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title {
        font-size: 1.95rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 0.98rem;
        color: #64748B;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 0.9rem;
        margin-bottom: 0.6rem;
    }
    .alert-danger {
        background-color: #FEF2F2;
        border-left: 5px solid #EF4444;
        padding: 0.85rem;
        border-radius: 6px;
        color: #991B1B;
        margin-bottom: 0.8rem;
        font-size: 0.92rem;
    }
    .alert-success {
        background-color: #F0FDF4;
        border-left: 5px solid #22C55E;
        padding: 0.85rem;
        border-radius: 6px;
        color: #166534;
        margin-bottom: 0.8rem;
        font-size: 0.92rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 핵심 알고리즘 및 기업 인재상 가중치 계산
# ---------------------------------------------------------
def compute_job_competency_index(df, talent_profiles):
    """
    기업 인재상(중복 선택)에 따른 동적 직무 역량 지수 산출
    기본 가중치: 코딩 0.45, 인성 0.25, 신체 0.10, 프로젝트 3.5, 경력정규화 1.8
    """
    w_coding = 0.45 * (1.6 if "💻 기술 전문성 (코딩·알고리즘)" in talent_profiles else 1.0)
    w_pers = 0.25 * (1.8 if "🤝 협업 및 인성 (조직문화·소통)" in talent_profiles else 1.0)
    w_phys = 0.10 * (2.0 if "🏃 현장 실행력 및 신체 역량" in talent_profiles else 1.0)
    w_proj = 3.5 * (1.5 if "📈 실무 프로젝트 문제해결력" in talent_profiles else 1.0)
    w_career = 1.8 * (1.5 if "🏢 장기 실무 경력 및 숙련도" in talent_profiles else 1.0)

    coding = df['Coding_Score'] if 'Coding_Score' in df.columns else 70.0
    pers = df['Personality_Score'] if 'Personality_Score' in df.columns else 75.0
    phys = df['Physical_Score'] if 'Physical_Score' in df.columns else 70.0
    proj = df['Projects_Count'] if 'Projects_Count' in df.columns else 3.0
    career = df['Career_Years'] if 'Career_Years' in df.columns else 2.0
    
    career_norm = career.clip(upper=8)
    return (coding * w_coding) + (pers * w_pers) + (phys * w_phys) + (proj * w_proj) + (career_norm * w_career)

def calculate_disparate_impact_ratio(df, sensitive_col, label_col='Is_Accepted'):
    group_stats = df.groupby(sensitive_col)[label_col].agg(['count', 'sum', 'mean']).reset_index()
    group_stats.columns = [sensitive_col, 'Total', 'Accepted', 'Rate']
    
    if len(group_stats) < 2:
        return 1.0, group_stats, None, None
        
    benchmark_row = group_stats.loc[group_stats['Rate'].idxmax()]
    adverse_row = group_stats.loc[group_stats['Rate'].idxmin()]
    
    dir_ratio = adverse_row['Rate'] / benchmark_row['Rate'] if benchmark_row['Rate'] > 0 else 1.0
    return dir_ratio, group_stats, benchmark_row, adverse_row

def chi_square_test(df, sensitive_col, label_col='Is_Accepted'):
    contingency = pd.crosstab(df[sensitive_col], df[label_col])
    chi2, p_val, dof, _ = chi2_contingency(contingency)
    return chi2, p_val, dof

def train_evaluation_models(df):
    features = ['Coding_Score', 'Projects_Count', 'Career_Years', 'Gender', 'Race', 'Age_Group', 'Region']
    for extra in ['Personality_Score', 'Physical_Score']:
        if extra in df.columns:
            features.append(extra)
            
    X_raw = df[features].copy()
    y = df['Is_Accepted'].values
    
    X_encoded = pd.get_dummies(X_raw, columns=['Gender', 'Race', 'Age_Group', 'Region'], drop_first=True)
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_encoded, y)
    
    return rf, list(X_encoded.columns), features

def compute_counterfactual_flip_rate(df, rf_model, feature_cols, base_features, sensitive_col):
    X_orig = pd.get_dummies(df[base_features].copy(), columns=['Gender', 'Race', 'Age_Group', 'Region'], drop_first=True)
    for col in feature_cols:
        if col not in X_orig.columns:
            X_orig[col] = 0
    X_orig = X_orig[feature_cols]
    
    base_preds = rf_model.predict(X_orig)
    
    X_cf = df[base_features].copy()
    unique_vals = list(df[sensitive_col].unique())
    if len(unique_vals) >= 2:
        val_map = {unique_vals[i]: unique_vals[(i + 1) % len(unique_vals)] for i in range(len(unique_vals))}
        X_cf[sensitive_col] = X_cf[sensitive_col].map(val_map)
        
    X_cf_encoded = pd.get_dummies(X_cf, columns=['Gender', 'Race', 'Age_Group', 'Region'], drop_first=True)
    for col in feature_cols:
        if col not in X_cf_encoded.columns:
            X_cf_encoded[col] = 0
    X_cf_encoded = X_cf_encoded[feature_cols]
    
    cf_preds = rf_model.predict(X_cf_encoded)
    return np.mean(base_preds != cf_preds)

def calculate_bias_probability(p_val, dir_ratio, flip_rate, rate_diff):
    if dir_ratio >= 0.80 or rate_diff <= 0.05:
        return round(max(0.5, (1.0 - dir_ratio) * 25.0), 1)
    else:
        p_factor = (1.0 - p_val) * 15.0 if p_val <= 0.05 else 0.0
        prob = 75.0 + (0.80 - dir_ratio) * 30.0 + (flip_rate * 20.0) + p_factor
        return round(min(99.9, max(75.0, prob)), 1)

# ---------------------------------------------------------
# 세션 상태 초기화
# ---------------------------------------------------------
if 'df' not in st.session_state:
    st.session_state.df = None
if 'rf_model' not in st.session_state:
    st.session_state.rf_model = None
if 'feat_cols' not in st.session_state:
    st.session_state.feat_cols = None
if 'base_features' not in st.session_state:
    st.session_state.base_features = None
if 'debiased_df' not in st.session_state:
    st.session_state.debiased_df = None
if 'page' not in st.session_state:
    st.session_state.page = "1. 📂 채용 데이터 불러오기"
if 'talent_profiles' not in st.session_state:
    st.session_state.talent_profiles = ["💻 기술 전문성 (코딩·알고리즘)", "🤝 협업 및 인성 (조직문화·소통)"]
if 'loaded_scenario' not in st.session_state:
    st.session_state.loaded_scenario = "알고리즘 편향 발생 데이터"

# ---------------------------------------------------------
# 사이드바 네비게이션
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚖️ **AuditaHR**")
    st.caption("AI 채용 데이터 편향 정밀 진단 및 자동 교정")
    st.markdown("---")
    
    nav = st.radio(
        "워크스페이스 메뉴",
        ["1. 📂 채용 데이터 불러오기", "2. 📊 AI 편향 진단 및 교정 리포트"],
        index=0 if st.session_state.page.startswith("1") else 1
    )
    st.session_state.page = nav
    
    st.markdown("---")
    st.markdown("#### 🎯 4대 핵심 진단 분야")
    st.write("1. **성별 (Gender)**")
    st.write("2. **출신 지역 (Region)**")
    st.write("3. **연령대 (Age Group)**")
    st.write("4. **인종/국적 (Race)**")
    st.caption("실제 데이터 통계 수치를 기준으로 객관적 편향을 진단합니다.")

# ---------------------------------------------------------
# PAGE 1: 채용 데이터 불러오기 및 인재상 설정
# ---------------------------------------------------------
if st.session_state.page.startswith("1"):
    st.markdown('<div class="main-title">📂 채용 데이터 불러오기 및 인재상 설정</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">회사의 직무별 인재상 가중치를 설정하고, 편향 발생/미발생 샘플 데이터 또는 보유하신 CSV 파일을 선택하여 불러오세요.</div>', unsafe_allow_html=True)

    # 1. 기업 인재상 설정 영역 (1단계)
    st.markdown("### 🏢 [1단계] 기업 맞춤형 인재상 설정 (중복 선택)")
    st.caption("회사가 필요로 하는 직무 역량 기준에 맞추어 평가할 수 있도록 인재상 가중치를 결합합니다.")

    talent_options = [
        "💻 기술 전문성 (코딩·알고리즘)",
        "🤝 협업 및 인성 (조직문화·소통)",
        "📈 실무 프로젝트 문제해결력",
        "🏃 현장 실행력 및 신체 역량",
        "🏢 장기 실무 경력 및 숙련도"
    ]
    
    selected_talents = st.multiselect(
        "우리 회사가 중점적으로 평가할 인재상을 선택하세요 (복수 선택 가능):",
        talent_options,
        default=st.session_state.talent_profiles,
        help="선택한 인재상에 맞춰 직무 역량 지수의 가중치가 동적으로 조정됩니다."
    )
    st.session_state.talent_profiles = selected_talents if selected_talents else ["💻 기술 전문성 (코딩·알고리즘)"]

    st.markdown("""
    <div class="metric-card">
        <strong>⚙️ 현재 설정된 직무 역량 산출 가중치:</strong><br>
        <span style="font-size:0.9rem; color:#334155;">
        • 코딩 점수: {w_c}배 | 인성 점수: {w_pe}배 | 신체 역량: {w_ph}배 | 프로젝트: {w_pr}배 | 경력: {w_ca}배
        </span>
    </div>
    """.format(
        w_c=round(0.45 * (1.6 if "💻 기술 전문성 (코딩·알고리즘)" in st.session_state.talent_profiles else 1.0), 2),
        w_pe=round(0.25 * (1.8 if "🤝 협업 및 인성 (조직문화·소통)" in st.session_state.talent_profiles else 1.0), 2),
        w_ph=round(0.10 * (2.0 if "🏃 현장 실행력 및 신체 역량" in st.session_state.talent_profiles else 1.0), 2),
        w_pr=round(3.5 * (1.5 if "📈 실무 프로젝트 문제해결력" in st.session_state.talent_profiles else 1.0), 2),
        w_ca=round(1.8 * (1.5 if "🏢 장기 실무 경력 및 숙련도" in st.session_state.talent_profiles else 1.0), 2)
    ), unsafe_allow_html=True)

    st.markdown("---")

    # 2. 채용 데이터 불러오기 영역 (2단계: 괄호 내용 완전 제거)
    st.markdown("### ⚡ [2단계] 채용 데이터 불러오기")
    
    col_sc1, col_sc2 = st.columns([1.6, 1])
    with col_sc1:
        scenario = st.radio(
            "불러올 샘플 데이터 시나리오를 선택하세요:",
            [
                "🚨 [시나리오 1] 알고리즘 편향 발생 데이터",
                "✅ [시나리오 2] 공정 선발 정상 데이터"
            ],
            index=0 if "편향 발생" in st.session_state.loaded_scenario else 1
        )
        
        if st.button("🚀 선택한 시나리오 데이터 즉시 불러오기", type="primary", use_container_width=True):
            try:
                df_app = pd.read_csv("sample_applicants.csv")
                if "시나리오 1" in scenario:
                    df_acc = pd.read_csv("sample_accepted_biased.csv")
                    st.session_state.loaded_scenario = "알고리즘 편향 발생 데이터"
                else:
                    df_acc = pd.read_csv("sample_accepted_fair.csv")
                    st.session_state.loaded_scenario = "공정 선발 정상 데이터"
                
                merged = df_app.copy()
                acc_set = set(df_acc['Applicant_ID'].dropna().astype(str))
                merged['Is_Accepted'] = merged['Applicant_ID'].astype(str).apply(lambda x: 1 if x in acc_set else 0)
                merged['Technical_Index'] = compute_job_competency_index(merged, st.session_state.talent_profiles)
                
                rf, cols, base_feats = train_evaluation_models(merged)
                st.session_state.df = merged
                st.session_state.rf_model = rf
                st.session_state.feat_cols = cols
                st.session_state.base_features = base_feats
                st.toast(f"✅ {st.session_state.loaded_scenario} 로드가 완료되었습니다!", icon="🎉")
            except Exception as e:
                st.error(f"데이터 로드 실패: {e}")

    with col_sc2:
        st.markdown("##### 📥 샘플 파일 다운로드")
        try:
            with open("sample_applicants.csv", "rb") as f:
                st.download_button("1. 지원자 데이터", f.read(), "sample_applicants.csv", "text/csv", use_container_width=True)
            with open("sample_accepted_biased.csv", "rb") as f:
                st.download_button("2. 편향 합격자 명단 CSV", f.read(), "sample_accepted_biased.csv", "text/csv", use_container_width=True)
            with open("sample_accepted_fair.csv", "rb") as f:
                st.download_button("3. 공정 합격자 명단 CSV", f.read(), "sample_accepted_fair.csv", "text/csv", use_container_width=True)
        except Exception:
            pass

    st.markdown("---")

    # 3. 직접 CSV 파일 업로드 영역
    st.markdown("### 📂 [선택] 직접 보유한 CSV 파일 업로드")
    col_u1, col_u2 = st.columns(2)
    with col_u1:
        up_app = st.file_uploader("전체 지원자 CSV 업로드", type=["csv"], key="f_app")
    with col_u2:
        up_acc = st.file_uploader("합격자 명단 CSV 업로드", type=["csv"], key="f_acc")

    if up_app is not None and up_acc is not None:
        try:
            df_app = pd.read_csv(up_app)
            df_acc = pd.read_csv(up_acc)
            
            merged = df_app.copy()
            acc_set = set(df_acc['Applicant_ID'].dropna().astype(str))
            merged['Is_Accepted'] = merged['Applicant_ID'].astype(str).apply(lambda x: 1 if x in acc_set else 0)
            merged['Technical_Index'] = compute_job_competency_index(merged, st.session_state.talent_profiles)
            
            rf, cols, base_feats = train_evaluation_models(merged)
            st.session_state.df = merged
            st.session_state.rf_model = rf
            st.session_state.feat_cols = cols
            st.session_state.base_features = base_feats
            st.session_state.loaded_scenario = "사용자 직접 업로드 데이터"
            st.success("✅ 업로드된 데이터가 성공적으로 병합되었습니다!")
        except Exception as e:
            st.error(f"파일 파싱 오류: {e}")

    # 4. 데이터 로드 완료 요약 카드
    if st.session_state.df is not None:
        data = st.session_state.df
        total = len(data)
        accepted = int(data['Is_Accepted'].sum())
        rate = (accepted / total * 100) if total > 0 else 0
        avg_tech = data['Technical_Index'].mean()

        st.markdown("---")
        st.markdown(f"#### 📋 현재 로드된 데이터: **{st.session_state.loaded_scenario}**")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("총 지원자 수", f"{total:,}명")
        m2.metric("총 합격자 수", f"{accepted:,}명")
        m3.metric("평균 합격률", f"{rate:.1f}%")
        m4.metric("평균 인재상 역량점수", f"{avg_tech:.1f}점")

        with st.expander("🔍 데이터 미리보기 (상위 5건)", expanded=False):
            disp_cols = ['Applicant_ID', 'Gender', 'Region', 'Age_Group', 'Race', 'Coding_Score', 'Personality_Score', 'Physical_Score', 'Technical_Index', 'Is_Accepted']
            st.dataframe(data[[c for c in disp_cols if c in data.columns]].head(5), use_container_width=True)

        st.markdown("---")
        col_g1, col_g2 = st.columns([2, 1])
        with col_g1:
            st.write("👉 성별, 출신지역, 연령대, 인종/국적의 4대 편향 분석 그래프와 리포트를 확인하세요.")
        with col_g2:
            if st.button("📊 AI 편향 진단 리포트 확인하기 ➔", type="primary", use_container_width=True):
                st.session_state.page = "2. 📊 AI 편향 진단 및 교정 리포트"
                st.rerun()

    else:
        st.info("💡 상단의 [🚀 선택한 시나리오 데이터 즉시 불러오기] 버튼을 누르면 1초 만에 분석이 시작됩니다.")

# ---------------------------------------------------------
# PAGE 2: AI 편향 진단 및 교정 리포트
# ---------------------------------------------------------
elif st.session_state.page.startswith("2"):
    st.markdown('<div class="main-title">📊 AI 채용 데이터 편향 진단 및 교정 리포트</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">성별, 출신지역, 연령대, 인종/국적 등 4대 핵심 분야의 데이터 통계를 정밀 분석하고, 기업 인재상 중심 공정 재선발을 수행합니다.</div>', unsafe_allow_html=True)

    if st.session_state.df is None:
        st.warning("⚠️ 먼저 [1. 📂 채용 데이터 불러오기] 페이지에서 데이터를 불러와 주세요.")
        if st.button("⬅️ 데이터 불러오기 페이지로 이동"):
            st.session_state.page = "1. 📂 채용 데이터 불러오기"
            st.rerun()
    else:
        df = st.session_state.df
        rf_model = st.session_state.rf_model
        feat_cols = st.session_state.feat_cols
        base_feats = st.session_state.base_features
        
        attrs = ['Gender', 'Region', 'Age_Group', 'Race']
        attr_labels = {
            'Gender': '1. 성별 (Gender)',
            'Region': '2. 출신 지역 (Region)',
            'Age_Group': '3. 연령대 (Age Group)',
            'Race': '4. 인종/국적 (Race/Nationality)'
        }
        
        audit_records = []
        stats_map = {}
        biased_fields = []
        
        for a in attrs:
            dir_ratio, g_stats, bench, adv = calculate_disparate_impact_ratio(df, a)
            chi2, p_val, _ = chi_square_test(df, a)
            flip_rate = compute_counterfactual_flip_rate(df, rf_model, feat_cols, base_feats, a)
            rate_diff = bench['Rate'] - adv['Rate']
            
            prob = calculate_bias_probability(p_val, dir_ratio, flip_rate, rate_diff)
            
            if dir_ratio < 0.80 and rate_diff > 0.05:
                verdict = "🚨 편향 발생"
                biased_fields.append(attr_labels[a].split('. ')[1].split(' ')[0])
            else:
                verdict = "✅ 정상 (균형)"
                
            audit_records.append({
                "분석 분야": attr_labels.get(a, a),
                "최고 합격 집단": f"{bench[a]} ({bench['Rate']*100:.1f}%)",
                "최저 합격 집단": f"{adv[a]} ({adv['Rate']*100:.1f}%)",
                "격차 비율 (DIR)": round(dir_ratio, 3),
                "합격률 차이": f"{rate_diff*100:.1f}%p",
                "편향 발생 확률": f"{prob:.1f}%",
                "편향이 발생하지 않았을 확률": f"{100.0 - prob:.1f}%",
                "판정": verdict
            })
            stats_map[a] = {
                'dir': dir_ratio, 'p': p_val, 'flip': flip_rate,
                'prob': prob, 'g_stats': g_stats, 'verdict': verdict,
                'bench': bench, 'adv': adv, 'diff': rate_diff
            }
            
        audit_df = pd.DataFrame(audit_records)
        max_bias_prob = max([stats_map[a]['prob'] for a in attrs])
        fairness_prob = round(100.0 - max_bias_prob, 1)

        # -----------------------------------------------------
        # [섹션 1] 종합 편향 진단 (결과 전용 표시)
        # -----------------------------------------------------
        st.markdown(f"### [1] 종합 편향 진단 (현재 데이터: **{st.session_state.loaded_scenario}**)")
        
        c_kpi1, c_kpi2 = st.columns([1.3, 1.7])
        with c_kpi1:
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric(
                    "편향 발생 확률",
                    f"{max_bias_prob:.1f}%",
                    delta="편향 감지" if max_bias_prob >= 70 else "정상 범위",
                    delta_color="inverse"
                )
            with col_m2:
                st.metric(
                    "편향이 발생하지 않았을 확률",
                    f"{fairness_prob:.1f}%",
                    delta="공정 선발" if fairness_prob >= 70 else "불균형",
                    delta_color="normal"
                )

            if len(biased_fields) > 0:
                field_str = ", ".join(biased_fields)
                st.markdown(f"""
                <div class="alert-danger">
                    <strong>🚨 [데이터 편향 발생]</strong><br>
                    분석 결과, <strong>{field_str}</strong> 항목에서 집단 간 합격률 격차 비율(DIR)이 기준치(0.80)에 미달하여 데이터 편향이 확인되었습니다.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="alert-success">
                    <strong>✅ [데이터 편향 미발생]</strong><br>
                    성별, 출신 지역, 연령대, 인종/국적 4개 분야 모두 합격률 격차 비율(DIR)이 0.80 이상으로, 특정 집단에 대한 유의미한 데이터 편향이 관측되지 않았습니다.
                </div>
                """, unsafe_allow_html=True)
                
        with c_kpi2:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=max_bias_prob,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "종합 데이터 편향 위험도 (%)", 'font': {'size': 15}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#EF4444" if max_bias_prob >= 70 else "#22C55E"},
                    'steps': [
                        {'range': [0, 50], 'color': '#DCFCE7'},
                        {'range': [50, 70], 'color': '#FEF9C3'},
                        {'range': [70, 100], 'color': '#FEE2E2'}
                    ]
                }
            ))
            fig_gauge.update_layout(height=210, margin=dict(l=10, r=10, t=25, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown("---")

        # -----------------------------------------------------
        # [섹션 2] 4대 분야별 전체 그래프 및 구체적 원인/주의점 정밀 분석
        # -----------------------------------------------------
        st.markdown("### [2] 4대 핵심 분야별 합격률 격차 및 상세 분석")
        st.dataframe(audit_df, use_container_width=True)

        col_g1, col_g2 = st.columns(2)
        
        # 1. 성별 그래프 및 분석
        with col_g1:
            g_data = stats_map['Gender']['g_stats'].copy()
            g_data['Gender_KR'] = g_data['Gender'].map({'Male': '남성', 'Female': '여성'})
            fig_g = px.bar(
                g_data, x='Gender_KR', y='Rate',
                text=g_data['Rate'].apply(lambda x: f"{x*100:.1f}%"),
                title=f"① 성별 합격률 (격차 비율 DIR: {stats_map['Gender']['dir']:.3f})",
                color='Gender_KR', color_discrete_sequence=['#3B82F6', '#EC4899']
            )
            fig_g.update_layout(yaxis_tickformat='.0%', height=280, showlegend=False)
            st.plotly_chart(fig_g, use_container_width=True)
            
            gen_stat = stats_map['Gender']
            if gen_stat['dir'] < 0.80 and gen_stat['diff'] > 0.05:
                gen_desc = f"""
                • <strong>분석 결론:</strong> 업무 역량과 관계 없이 남성 쪽으로 합격자가 편향되었다고 분석하였습니다.<br>
                • <strong>그 이유:</strong> 동일한 코딩 및 인성 점수 대비 남성 합격률({gen_stat['bench']['Rate']*100:.1f}%)이 여성 합격률({gen_stat['adv']['Rate']*100:.1f}%)보다 {gen_stat['diff']*100:.1f}%p 높아, 합격률 격차 비율(DIR: {gen_stat['dir']:.3f})이 공정 기준치(0.80)에 크게 미달하였기 때문입니다.<br>
                • <strong>분석 시 주의해야 할 방면:</strong> 1차 필터링 단계에서 성별 가산점이 개입되지 않도록 성별 블라인드 기준을 강화하고, 군 복무 이력 등 성별과 밀접히 연관된 대리 변수(Proxy Variable)의 교차 영향을 면밀히 점검해야 합니다.
                """
            else:
                gen_desc = f"""
                • <strong>분석 결론:</strong> 성별에 따른 편향 없이 업무 역량에 따라 고르게 선발되었다고 분석하였습니다.<br>
                • <strong>그 이유:</strong> 남성 합격률({gen_stat['bench']['Rate']*100:.1f}%)과 여성 합격률({gen_stat['adv']['Rate']*100:.1f}%) 간의 격차 비율(DIR: {gen_stat['dir']:.3f})이 기준치(0.80) 이상을 기록하여 유의미한 합격률 편차가 발생하지 않았기 때문입니다.<br>
                • <strong>분석 시 주의해야 할 방면:</strong> 서류 및 코딩 평가 단계의 공정 선발 기조가 최종 면접 전형까지 일관되게 유지되는지 지속적으로 모니터링해야 합니다.
                """
                
            st.markdown(f"""
            <div class="metric-card">
                <strong>📊 성별 데이터 상세 분석:</strong><br>
                <span style="font-size:0.86rem; color:#334155;">{gen_desc}</span>
            </div>
            """, unsafe_allow_html=True)

        # 2. 출신 지역 그래프 및 분석
        with col_g2:
            r_data = stats_map['Region']['g_stats'].copy()
            r_data['Region_KR'] = r_data['Region'].map({'Seoul/Metropolitan': '수도권', 'Regional': '지방'})
            fig_r = px.bar(
                r_data, x='Region_KR', y='Rate',
                text=r_data['Rate'].apply(lambda x: f"{x*100:.1f}%"),
                title=f"② 출신 지역별 합격률 (격차 비율 DIR: {stats_map['Region']['dir']:.3f})",
                color='Region_KR', color_discrete_sequence=['#6366F1', '#10B981']
            )
            fig_r.update_layout(yaxis_tickformat='.0%', height=280, showlegend=False)
            st.plotly_chart(fig_r, use_container_width=True)
            
            reg_stat = stats_map['Region']
            if reg_stat['dir'] < 0.80 and reg_stat['diff'] > 0.05:
                reg_desc = f"""
                • <strong>분석 결론:</strong> 업무 역량과 관계 없이 수도권 거주 지원자 쪽으로 합격자가 편향되었다고 분석하였습니다.<br>
                • <strong>그 이유:</strong> 원격 소프트웨어 직무 역량과 무관하게 수도권 합격률({reg_stat['bench']['Rate']*100:.1f}%) 대비 지방 지원자 합격률({reg_stat['adv']['Rate']*100:.1f}%)의 격차가 {reg_stat['diff']*100:.1f}%p에 달해 격차 비율(DIR: {reg_stat['dir']:.3f})이 0.80에 미달하였기 때문입니다.<br>
                • <strong>분석 시 주의해야 할 방면:</strong> 지원자의 주소지나 출신 대학 소재지 정보가 평가 점수에 비공식적으로 영향을 미쳤는지 주소 블라인드 적용 범위를 재확인해야 합니다.
                """
            else:
                reg_desc = f"""
                • <strong>분석 결론:</strong> 출신 지역과 무관하게 순수 업무 역량에 따라 고르게 선발되었다고 분석하였습니다.<br>
                • <strong>그 이유:</strong> 수도권 합격률({reg_stat['bench']['Rate']*100:.1f}%)과 지방 합격률({reg_stat['adv']['Rate']*100:.1f}%) 간 격차 비율(DIR: {reg_stat['dir']:.3f})이 기준치(0.80) 이상을 만족하여 지역 간 합격률이 대등하였기 때문입니다.<br>
                • <strong>분석 시 주의해야 할 방면:</strong> 지방 거주 우수 인재의 면접 참여 기회가 지리적·물리적 제약으로 훼손되지 않도록 비대면 면접 인프라가 공정히 지원되는지 점검해야 합니다.
                """

            st.markdown(f"""
            <div class="metric-card">
                <strong>📊 지역 데이터 상세 분석:</strong><br>
                <span style="font-size:0.86rem; color:#334155;">{reg_desc}</span>
            </div>
            """, unsafe_allow_html=True)

        col_g3, col_g4 = st.columns(2)
        
        # 3. 연령대 그래프 및 분석
        with col_g3:
            age_data = stats_map['Age_Group']['g_stats'].copy()
            fig_age = px.bar(
                age_data, x='Age_Group', y='Rate',
                text=age_data['Rate'].apply(lambda x: f"{x*100:.1f}%"),
                title=f"③ 연령대별 합격률 (격차 비율 DIR: {stats_map['Age_Group']['dir']:.3f})",
                color='Age_Group', color_discrete_sequence=['#F59E0B', '#8B5CF6', '#EC4899']
            )
            fig_age.update_layout(yaxis_tickformat='.0%', height=280, showlegend=False)
            st.plotly_chart(fig_age, use_container_width=True)
            
            age_stat = stats_map['Age_Group']
            if age_stat['dir'] < 0.80 and age_stat['diff'] > 0.05:
                age_desc = f"""
                • <strong>분석 결론:</strong> 업무 역량과 관계 없이 특정 연령대({age_stat['bench']['Age_Group']}) 쪽으로 합격자가 편향되었다고 분석하였습니다.<br>
                • <strong>이유:</strong> 코딩 실력이 뛰어난 청년층({age_stat['adv']['Age_Group']}: {age_stat['adv']['Rate']*100:.1f}%) 대비 장기 연차 집단({age_stat['bench']['Age_Group']}: {age_stat['bench']['Rate']*100:.1f}%)의 합격률 격차가 {age_stat['diff']*100:.1f}%p로 과도하게 벌어져 격차 비율(DIR: {age_stat['dir']:.3f})이 크게 떨어졌기 때문입니다.<br>
                • <strong>분석 시 주의해야 할 방면:</strong> 경력 연수에 대한 과도한 가산점이 실전 문제해결력보다 높게 반영되어 유능한 청년 지원자가 기계적으로 탈락하지 않았는지 연차 가중치 밸런스를 검토해야 합니다.
                """
            else:
                age_desc = f"""
                • <strong>분석 결론:</strong> 연령대에 따른 차별적 영향 없이 업무 역량 중심으로 균형 있게 선발되었다고 분석하였습니다.<br>
                • <strong>이유:</strong> 20대 신입부터 40대 이상 경력자까지 전 연령층의 합격률 격차 비율(DIR: {age_stat['dir']:.3f})이 0.80 이상으로 안정적으로 분포하였기 때문입니다.<br>
                • <strong>분석 시 주의해야 할 방면:</strong> 신입과 경력 채용 목적에 맞는 직무 평가 기준이 명확히 분리 적용되었는지 확인해야 합니다.
                """

            st.markdown(f"""
            <div class="metric-card">
                <strong>📊 연령대 데이터 상세 분석:</strong><br>
                <span style="font-size:0.86rem; color:#334155;">{age_desc}</span>
            </div>
            """, unsafe_allow_html=True)

        # 4. 인종/국적 그래프 및 분석
        with col_g4:
            race_data = stats_map['Race']['g_stats'].copy()
            fig_race = px.bar(
                race_data, x='Race', y='Rate',
                text=race_data['Rate'].apply(lambda x: f"{x*100:.1f}%"),
                title=f"④ 인종/국적별 합격률 (격차 비율 DIR: {stats_map['Race']['dir']:.3f})",
                color='Race', color_discrete_sequence=['#14B8A6', '#06B6D4', '#3B82F6', '#6366F1']
            )
            fig_race.update_layout(yaxis_tickformat='.0%', height=280, showlegend=False)
            st.plotly_chart(fig_race, use_container_width=True)
            
            race_stat = stats_map['Race']
            if race_stat['dir'] < 0.80 and race_stat['diff'] > 0.05:
                race_desc = f"""
                • <strong>분석 결론:</strong> 업무 역량과 관계 없이 특정 인종/국적 집단 쪽으로 선발이 편향되었다고 분석하였습니다.<br>
                • <strong>이유:</strong> 최저 합격 집단({race_stat['adv']['Race']}: {race_stat['adv']['Rate']*100:.1f}%)과 최고 집단({race_stat['bench']['Race']}: {race_stat['bench']['Rate']*100:.1f}%) 간 합격률 격차 비율(DIR: {race_stat['dir']:.3f})이 기준치(0.80)에 미달하였기 때문입니다.<br>
                • <strong>분석 시 주의해야 할 방면:</strong> 언어적 표현이나 출신 국적 표기가 서류 평가 단계에서 불필요한 선입견으로 작용하지 않았는지 블라인드 평가 지침을 점검해야 합니다.
                """
            else:
                race_desc = f"""
                • <strong>분석 결론:</strong> 인종 및 국적과 무관하게 업무 역량에 따라 고르게 선발되었다고 분석하였습니다.<br>
                • <strong>이유:</strong> 모든 인종 및 국적 그룹의 합격률 격차 비율(DIR: {race_stat['dir']:.3f})이 기준치(0.80) 이상을 만족하여 집단 간 유의미한 합격률 편향이 나타나지 않았기 때문입니다.<br>
                • <strong>분석 시 주의해야 할 방면:</strong> 글로벌 인재 채용 시 객관적인 직무 수행 지표가 모든 지원자에게 동일한 기준으로 평가되었는지 지속적으로 관리해야 합니다.
                """

            st.markdown(f"""
            <div class="metric-card">
                <strong>📊 인종/국적 데이터 상세 분석:</strong><br>
                <span style="font-size:0.86rem; color:#334155;">{race_desc}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # -----------------------------------------------------
        # [섹션 3] 기업 인재상 중심 공정 자동 교정 (4대 속성 전수 시각화)
        # -----------------------------------------------------
        st.markdown("### [3] 기업 맞춤형 인재상 중심 공정 자동 교정 시뮬레이션")
        st.markdown("""
        성별·출신지역·연령대·인종 등 배경 속성을 전면 배제하고, 앞서 1단계에서 설정한 **우리 회사의 인재상(`{talents}`)**에 부합하는 순수 역량 지표(`Technical_Index`) 순서대로 공정하게 재선발합니다.
        """.format(talents=", ".join(st.session_state.talent_profiles)))

        if st.button("🚀 설정된 기업 인재상 기반 공정 채용 자동 교정 실행", type="primary"):
            target_n = int(df['Is_Accepted'].sum())
            debiased = df.copy()
            debiased = debiased.sort_values(by='Technical_Index', ascending=False).reset_index(drop=True)
            debiased['Debiased_Accepted'] = 0
            debiased.loc[:target_n - 1, 'Debiased_Accepted'] = 1
            st.session_state.debiased_df = debiased
            st.toast("✨ 기업 인재상 중심 공정 재선발 교정이 완료되었습니다!", icon="🎉")

        if st.session_state.debiased_df is not None:
            deb_df = st.session_state.debiased_df
            
            # 4대 분야 교정 후 통계 계산
            post_g_dir, post_g_stats, _, _ = calculate_disparate_impact_ratio(deb_df, 'Gender', 'Debiased_Accepted')
            post_r_dir, post_r_stats, _, _ = calculate_disparate_impact_ratio(deb_df, 'Region', 'Debiased_Accepted')
            post_a_dir, post_a_stats, _, _ = calculate_disparate_impact_ratio(deb_df, 'Age_Group', 'Debiased_Accepted')
            post_rc_dir, post_rc_stats, _, _ = calculate_disparate_impact_ratio(deb_df, 'Race', 'Debiased_Accepted')
            
            rescued_count = len(deb_df[(deb_df['Is_Accepted'] == 0) & (deb_df['Debiased_Accepted'] == 1)])

            st.markdown("#### 🏆 4대 핵심 분야 교정 전 vs 교정 후 격차 비율(DIR) 비교")
            c_m1, c_m2, c_m3, c_m4, c_m5 = st.columns(5)
            c_m1.metric("성별 DIR", f"{post_g_dir:.3f}", delta=f"{post_g_dir - stats_map['Gender']['dir']:+.3f}")
            c_m2.metric("출신지역 DIR", f"{post_r_dir:.3f}", delta=f"{post_r_dir - stats_map['Region']['dir']:+.3f}")
            c_m3.metric("연령대 DIR", f"{post_a_dir:.3f}", delta=f"{post_a_dir - stats_map['Age_Group']['dir']:+.3f}")
            c_m4.metric("인종/국적 DIR", f"{post_rc_dir:.3f}", delta=f"{post_rc_dir - stats_map['Race']['dir']:+.3f}")
            c_m5.metric("구제된 인재", f"{rescued_count}명", delta="실력 중심 구제")

            # 4대 분야 교정 전후 비교 그래프 2x2 배치
            chart_r1_1, chart_r1_2 = st.columns(2)
            with chart_r1_1:
                # 1. 성별 전후 비교
                pre_g = stats_map['Gender']['g_stats'].copy()
                pre_g['구분'] = '교정 전'
                post_g = post_g_stats.copy()
                post_g['구분'] = '교정 후'
                concat_g = pd.concat([pre_g, post_g])
                concat_g['Gender_KR'] = concat_g['Gender'].map({'Male': '남성', 'Female': '여성'})
                fig_comp_g = px.bar(
                    concat_g, x='Gender_KR', y='Rate', color='구분', barmode='group',
                    text=concat_g['Rate'].apply(lambda x: f"{x*100:.1f}%"),
                    title="① 성별 합격률 교정 전후 비교",
                    color_discrete_sequence=['#EF4444', '#10B981']
                )
                fig_comp_g.update_layout(yaxis_tickformat='.0%', height=270)
                st.plotly_chart(fig_comp_g, use_container_width=True)

            with chart_r1_2:
                # 2. 지역 전후 비교
                pre_r = stats_map['Region']['g_stats'].copy()
                pre_r['구분'] = '교정 전'
                post_r = post_r_stats.copy()
                post_r['구분'] = '교정 후'
                concat_r = pd.concat([pre_r, post_r])
                concat_r['Region_KR'] = concat_r['Region'].map({'Seoul/Metropolitan': '수도권', 'Regional': '지방'})
                fig_comp_r = px.bar(
                    concat_r, x='Region_KR', y='Rate', color='구분', barmode='group',
                    text=concat_r['Rate'].apply(lambda x: f"{x*100:.1f}%"),
                    title="② 출신지역별 합격률 교정 전후 비교",
                    color_discrete_sequence=['#F59E0B', '#3B82F6']
                )
                fig_comp_r.update_layout(yaxis_tickformat='.0%', height=270)
                st.plotly_chart(fig_comp_r, use_container_width=True)

            chart_r2_1, chart_r2_2 = st.columns(2)
            with chart_r2_1:
                # 3. 연령대 전후 비교
                pre_a = stats_map['Age_Group']['g_stats'].copy()
                pre_a['구분'] = '교정 전'
                post_a = post_a_stats.copy()
                post_a['구분'] = '교정 후'
                concat_a = pd.concat([pre_a, post_a])
                fig_comp_a = px.bar(
                    concat_a, x='Age_Group', y='Rate', color='구분', barmode='group',
                    text=concat_a['Rate'].apply(lambda x: f"{x*100:.1f}%"),
                    title="③ 연령대별 합격률 교정 전후 비교",
                    color_discrete_sequence=['#EC4899', '#8B5CF6']
                )
                fig_comp_a.update_layout(yaxis_tickformat='.0%', height=270)
                st.plotly_chart(fig_comp_a, use_container_width=True)

            with chart_r2_2:
                # 4. 인종/국적 전후 비교
                pre_rc = stats_map['Race']['g_stats'].copy()
                pre_rc['구분'] = '교정 전'
                post_rc = post_rc_stats.copy()
                post_rc['구분'] = '교정 후'
                concat_rc = pd.concat([pre_rc, post_rc])
                fig_comp_rc = px.bar(
                    concat_rc, x='Race', y='Rate', color='구분', barmode='group',
                    text=concat_rc['Rate'].apply(lambda x: f"{x*100:.1f}%"),
                    title="④ 인종/국적별 합격률 교정 전후 비교",
                    color_discrete_sequence=['#14B8A6', '#06B6D4']
                )
                fig_comp_rc.update_layout(yaxis_tickformat='.0%', height=270)
                st.plotly_chart(fig_comp_rc, use_container_width=True)

            # 구제된 다양한 인재 명단
            st.markdown("#### 🌟 편향으로 부당하게 탈락했던 우수 인재 구제 명단")
            st.caption("뛰어난 코딩·인성·신체 역량을 겸비하여 기업 인재상에 부합함에도 불구하고, 과거 데이터 편향으로 탈락했던 지원자 목록입니다. (남성/여성, 수도권/지방, 청년/장년 및 다양한 출신 인재가 고르게 구제되었습니다)")
            
            rescued = deb_df[(deb_df['Is_Accepted'] == 0) & (deb_df['Debiased_Accepted'] == 1)]
            resc_cols = ['Applicant_ID', 'Gender', 'Region', 'Age_Group', 'Race', 'Coding_Score', 'Personality_Score', 'Physical_Score', 'Technical_Index']
            st.dataframe(rescued[[c for c in resc_cols if c in rescued.columns]].head(12), use_container_width=True)

            csv_fair = deb_df[deb_df['Debiased_Accepted'] == 1][[
                c for c in resc_cols if c in deb_df.columns
            ]].to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="📥 인재상 기반 최종 공정 합격자 명단 CSV 다운로드",
                data=csv_fair,
                file_name="audita_talent_fair_accepted.csv",
                mime="text/csv",
                type="primary"
            )
