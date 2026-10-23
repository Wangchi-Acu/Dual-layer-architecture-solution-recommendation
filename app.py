# -*- coding: utf-8 -*-
"""
app.py — 失眠针刺方案智能推荐系统（双层架构）Streamlit 入口

本地运行：  streamlit run app.py
线上部署：  推送本目录到 GitHub → share.streamlit.io → New app → 选择仓库与 app.py

正式版（v1.3）：第一层 RBF SVM 三分类（15 特征 + OvR Platt 校准），
第二层 极端随机树@3 / 梯度提升@8 / 极端随机树@11 个交互特征。
v1.2：第一层完成方案推荐后，第二层只展示推荐方案的显效概率与阈值判定，
并同步展示该方案的穴位定位示意图（动图）；不再展示三方案并列比较与两层一致性提示。
v1.3：按编码本更正方案命名（1=通督调卫、2=通督调脏、3=通督养心）；
变量标签按编码本补全（DI=合并疾病史、SS=伴随症状、SF=主诉、PA–PG=PSQI 成分）；
人口学分类变量改为带含义下拉框（如 0=小学及以下）。
"""
import os
import sys

import pandas as pd
import streamlit as st
import yaml

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from inference import load_assets, predict_one, predict_batch  # noqa: E402

# ---------------------------------------------------------------- 配置与资产
with open(os.path.join(PROJECT_ROOT, 'config.yaml'), encoding='utf-8') as f:
    CFG = yaml.safe_load(f)

st.set_page_config(page_title=CFG['app']['title'], page_icon='🩺', layout='wide')


@st.cache_resource(show_spinner='正在加载模型资产…')
def get_assets():
    return load_assets(PROJECT_ROOT)


@st.cache_data
def get_var_meta():
    return pd.read_csv(os.path.join(PROJECT_ROOT, CFG['assets_dir'], 'var_meta.csv'))


@st.cache_data
def get_example_patients():
    return pd.read_csv(os.path.join(PROJECT_ROOT, CFG['assets_dir'], 'example_patients.csv'))


try:
    ASSETS = get_assets()
except Exception as e:  # 自检失败：版本/特征清单不配套
    st.error(f'模型资产加载失败：{e}')
    st.stop()

SCHEMES = CFG['schemes']

# ---------------------------------------------------------------- 页面头部
st.title(CFG['app']['title'])
st.caption(f"模型版本：{CFG['app']['version']} ｜ 第一层：方案推荐（三分类，推荐取未校准 argmax） ｜ 第二层：推荐方案疗效预测 + 穴位定位示意")

with st.sidebar:
    st.header('功能')
    mode = st.radio('推理模式', ['单例录入', '批量上传'], index=0)
    st.divider()
    st.caption('模型版本')
    st.code(CFG['app']['version'], language=None)
    st.caption('模型自检：sklearn 版本 / 特征清单一致性 / 预测接口均已通过 ✓')


# ---------------------------------------------------------------- 结果渲染
def render_result(result: dict):
    l1, l2 = result['layer1'], result['layer2']

    st.subheader('第一层：方案推荐')
    rec = str(l1['recommend'])
    st.success(f"推荐方案：**{SCHEMES.get(rec, rec)}**（类别 {rec}，依据未校准模型得分 argmax）")
    st.caption('注：第一层推荐取未校准 softmax 得分的 argmax（与研究管线 OOF 评估口径一致）；'
               '下方校准后概率仅供把握度参考，不参与方案排序。')
    prob_df = pd.DataFrame(
        {'方案': [SCHEMES.get(str(c), str(c)) for c in l1['classes']],
         '校准后概率（把握度参考）': l1['proba'],
         '未校准得分（推荐依据）': l1['score_raw']}
    ).set_index('方案')
    st.bar_chart(prob_df, horizontal=True)

    st.subheader('第二层：推荐方案疗效预测')
    rec_target = f'y_bin{rec}'
    info = l2[rec_target]
    name = SCHEMES.get(rec, rec)
    col_prob, col_img = st.columns([1, 1])
    with col_prob:
        st.markdown(f"**{name} · 显效概率**")
        st.metric('显效概率', f"{info['proba']:.1%}")
        st.progress(min(max(info['proba'], 0.0), 1.0))
        verdict = '✅ 预测显效' if info['predict_effective'] else '❌ 预测未显效'
        st.caption(f"{verdict}（阈值 {info['threshold']:.3f}，来自该方案训练集 OOF Youden）")
        st.caption('说明：第二层仅输出第一层推荐方案对应的显效概率；三个二分类模型各自独立定标，'
                   '概率不可跨方案直接比较。')
    with col_img:
        img_rel = CFG.get('acupoint_images', {}).get(rec)
        img_path = os.path.join(PROJECT_ROOT, img_rel) if img_rel else None
        if img_path and os.path.exists(img_path):
            st.image(img_path, caption=f'{name}穴位定位示意（动图）', use_container_width=True)
        else:
            st.info('穴位定位示意图缺失：请将对应图片放入配置指定的 assets 路径。')


# ---------------------------------------------------------------- 单例模式
if mode == '单例录入':
    meta = get_var_meta()
    if 'form_values' not in st.session_state:
        st.session_state['form_values'] = {
            r['name']: (int(r['default']) if bool(r['is_int']) else float(r['default']))
            for _, r in meta.iterrows()}

    def load_example():
        ex = get_example_patients().iloc[0]
        for k in st.session_state['form_values']:
            if k in ex.index and pd.notna(ex[k]):
                st.session_state['form_values'][k] = ex[k]

    st.button('载入示例患者', on_click=load_example)

    with st.form('patient_form'):
        groups = meta['group'].unique().tolist()
        tabs = st.tabs(groups)
        for tab, g in zip(tabs, groups):
            with tab:
                sub = meta[meta['group'] == g]
                grid = st.columns(3)
                for i, (_, r) in enumerate(sub.iterrows()):
                    with grid[i % 3]:
                        cur = st.session_state['form_values'].get(r['name'], r['default'])
                        if bool(r['is_cat']):
                            opts = [int(x) for x in str(r['options']).split('|')]
                            try:
                                idx = opts.index(int(cur))
                            except (ValueError, TypeError):
                                idx = 0
                            # 选项含义映射（var_meta.csv option_labels 列，形如 "0=未婚|1=已婚"）
                            ol = r.get('option_labels')
                            lab = {}
                            if isinstance(ol, str) and ol.strip():
                                for part in ol.split('|'):
                                    if '=' in part:
                                        k, vlab = part.split('=', 1)
                                        lab[int(k.strip())] = vlab.strip()
                            v = st.selectbox(
                                r['label'], opts, index=idx,
                                format_func=(lambda m: (lambda x: f'{x}={m[x]}' if x in m else str(x)))(lab),
                                key=f"in_{r['name']}")
                            st.session_state['form_values'][r['name']] = int(v)
                        elif bool(r['is_int']):
                            v = st.number_input(r['label'], value=int(round(float(cur))), step=1,
                                                min_value=int(r['min']), max_value=int(r['max']),
                                                key=f"in_{r['name']}")
                            st.session_state['form_values'][r['name']] = int(v)
                        else:
                            v = st.number_input(r['label'], value=float(cur), step=0.01,
                                                format='%.4f', key=f"in_{r['name']}")
                            st.session_state['form_values'][r['name']] = float(v)
        submitted = st.form_submit_button('开始预测', type='primary', use_container_width=True)

    if submitted:
        raw = dict(st.session_state['form_values'])
        result = predict_one(ASSETS, raw)
        render_result(result)

# ---------------------------------------------------------------- 批量模式
else:
    st.markdown('上传 CSV（每行一名患者，列名为原始变量名）。缺失列/缺失值将自动填补（第二层按训练集中位数，第一层按训练时拟合的填补器）。')
    ex = get_example_patients()
    st.download_button('下载录入模板（含 3 行示例）',
                       ex.to_csv(index=False).encode('utf-8-sig'),
                       file_name='录入模板.csv', mime='text/csv')
    up = st.file_uploader('选择 CSV 文件', type=['csv'])
    if up is not None:
        try:
            raw_df = pd.read_csv(up)
        except UnicodeDecodeError:
            up.seek(0)
            raw_df = pd.read_csv(up, encoding='gbk')
        st.write(f'共读取 {len(raw_df)} 行。')
        if st.button('批量预测', type='primary'):
            out = predict_batch(ASSETS, raw_df)
            result_df = pd.concat([raw_df.reset_index(drop=True), out.reset_index(drop=True)], axis=1)
            st.dataframe(out, use_container_width=True)
            st.download_button('下载预测结果',
                               result_df.to_csv(index=False).encode('utf-8-sig'),
                               file_name='双层预测结果.csv', mime='text/csv')

# ---------------------------------------------------------------- 页脚
st.divider()
st.warning(CFG['app']['disclaimer'])
