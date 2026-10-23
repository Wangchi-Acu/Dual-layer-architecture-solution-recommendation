# -*- coding: utf-8 -*-
"""
inference.py — 资产加载与双层推理（不依赖 streamlit，可独立测试）

正式模型资产（models/ 目录，全部来自训练管线定稿产物）：
    layer1_multiclass.pkl   第一层：RBF SVM（BroadClassifier）三分类包
                            {model_name, n_features, features, params,
                             master_split_seed, model_random_seed, preprocess, model}
    layer1_calibrator.pkl   第一层 OvR sigmoid（Platt）校准器
                            {method, classes, calibrators, features, ...}
    layer2_y_bin1/2/3.pkl   第二层：三个方案的二分类自包含模型包
                            {model, features, fill_median, target, youden_threshold}
    feature_lists.json      {target: [特征名, ...]}（与 pkl 内清单一致，供自检与展示）
    train_medians.json      {target: {feature: 训练集中位数}}（第二层用；第一层由
                            pkl 内 preprocess 的拟合填补器完成，无需外部中位数）
    thresholds.json         {target: Youden 阈值（来自训练集 OOF）}

第一层推理路径（与研究管线 07_predict_calibrated.py 同源）：
    transform(frame, preprocess) → score_matrix（softmax 得分）
    → decision_function 逐类 sigmoid 校准 → 归一化得校准概率。
推荐口径（v1.1 修订）：推荐方案取**未校准得分的 argmax**。
    原因：本模型 decision_function 动态范围极窄（dev 上约 ±0.16），
    OvR sigmoid 校准以各类基率为锚，校准后 argmax 对全部患者恒为方案3，
    丧失区分能力；未校准 argmax 与研究管线 m.predict 逐例一致（dev 上 100%），
    与论文 OOF 评估口径一致。校准后概率仍输出，仅供把握度参考，不参与推荐。
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
import sklearn

import common  # noqa: F401  必须可导入：layer1_multiclass.pkl 按模块名引用 BroadClassifier
from common import transform, score_matrix
from features import resolve_feature

REQUIRED_SKLEARN_MAJOR_MINOR = (1, 7)  # pkl 训练环境的 sklearn 主版本
BIN_TARGETS = ['y_bin1', 'y_bin2', 'y_bin3']
CLS_TARGET = 'y_cls'
CAT_COLS = ['OCC', 'EDU']  # 第一层分类变量（独热编码，整型编码）


def load_assets(project_root: str) -> dict:
    mdir = os.path.join(project_root, 'models')
    with open(os.path.join(mdir, 'feature_lists.json'), encoding='utf-8') as f:
        feature_lists = json.load(f)
    with open(os.path.join(mdir, 'train_medians.json'), encoding='utf-8') as f:
        medians = json.load(f)
    with open(os.path.join(mdir, 'thresholds.json'), encoding='utf-8') as f:
        thresholds = json.load(f)

    assets = {
        'feature_lists': feature_lists,
        'medians': medians,
        'thresholds': {k: float(v) for k, v in thresholds.items()},
        'layer1': joblib.load(os.path.join(mdir, 'layer1_multiclass.pkl')),
        'layer1_calibrator': joblib.load(os.path.join(mdir, 'layer1_calibrator.pkl')),
        'layer2': {t: joblib.load(os.path.join(mdir, f'layer2_{t}.pkl')) for t in BIN_TARGETS},
    }
    self_check(assets)
    return assets


def self_check(assets: dict) -> None:
    """启动自检：版本、pkl 内清单与 JSON 一致性、接口可用性。失败应阻止应用启动。"""
    major, minor = sklearn.__version__.split('.')[:2]
    if (int(major), int(minor)) != REQUIRED_SKLEARN_MAJOR_MINOR:
        raise RuntimeError(
            f'sklearn 版本不匹配：当前 {sklearn.__version__}，'
            f'pkl 训练环境为 {REQUIRED_SKLEARN_MAJOR_MINOR[0]}.{REQUIRED_SKLEARN_MAJOR_MINOR[1]}.x。'
            f'请按 requirements.txt 安装对应版本。'
        )
    fl = assets['feature_lists']

    b1 = assets['layer1']
    if list(b1['features']) != list(fl[CLS_TARGET]):
        raise RuntimeError('layer1_multiclass.pkl 特征清单与 feature_lists.json 不一致。')
    if list(assets['layer1_calibrator']['features']) != list(b1['features']):
        raise RuntimeError('校准器与第一层模型的特征清单不一致。')
    if not hasattr(b1['model'], 'predict'):
        raise RuntimeError('第一层模型缺少 predict 接口。')

    for t in BIN_TARGETS:
        b = assets['layer2'][t]
        if list(b['features']) != list(fl[t]):
            raise RuntimeError(f'layer2_{t}.pkl 特征清单与 feature_lists.json 不一致。')
        if not hasattr(b['model'], 'predict_proba'):
            raise RuntimeError(f'{t} 模型不支持 predict_proba。')
        n_in = getattr(b['model'], 'n_features_in_', None)
        if n_in is not None and n_in != len(b['features']):
            raise RuntimeError(
                f'{t} 特征数不一致：pkl 需要 {n_in} 个，清单提供 {len(b["features"])} 个。'
            )
        if t not in assets['thresholds']:
            raise RuntimeError(f'thresholds.json 缺少 {t}。')


def build_feature_frame(raw_rows: list, feature_list: list, medians: dict) -> pd.DataFrame:
    """第二层用：raw_rows → 按 feature_list 顺序构造交互特征矩阵并填补训练集中位数。"""
    data = [[resolve_feature(f, raw) for f in feature_list] for raw in raw_rows]
    X = pd.DataFrame(data, columns=feature_list, dtype=float)
    for f in feature_list:
        med = medians.get(f)
        if med is not None:
            X[f] = X[f].fillna(float(med))
    return X


def _layer1_frame(raw_rows: list, features: list) -> pd.DataFrame:
    """第一层用：原始变量（含 OCC/EDU 分类编码）直接成框，预处理由 pkl 内状态完成。"""
    df = pd.DataFrame([{f: raw.get(f, np.nan) for f in features} for raw in raw_rows])
    for c in CAT_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype('Int64')
    num_cols = [c for c in df.columns if c not in CAT_COLS]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')
    return df


def _layer1_predict(frame: pd.DataFrame, bundle: dict, calibrator: dict):
    """返回 (校准概率矩阵, 未校准 softmax 得分矩阵)，与管线 07_predict_calibrated.py 同口径。"""
    Z = transform(frame[list(bundle['features'])], bundle['preprocess'])
    m = bundle['model']
    q = score_matrix(m, Z)
    if hasattr(m, 'decision_function'):
        s = np.asarray(m.decision_function(Z), float)
        if s.ndim == 1:
            s = np.c_[-s, s]
    else:
        s = np.log(np.clip(q, 1e-12, 1.0))
    p = np.zeros_like(q)
    for k, c in enumerate(calibrator['calibrators']):
        if calibrator['method'] == 'sigmoid':
            p[:, k] = c.predict_proba(s[:, k].reshape(-1, 1))[:, 1]
        else:
            p[:, k] = c.predict(s[:, k])
    p = np.clip(p, 1e-12, None)
    p = p / p.sum(axis=1, keepdims=True)
    return p, q


def predict_one(assets: dict, raw: dict) -> dict:
    """单例推理。返回第一层类别概率 + 第二层各方案有效概率与判定。"""
    fl, med, thr = assets['feature_lists'], assets['medians'], assets['thresholds']

    X1 = _layer1_frame([raw], fl[CLS_TARGET])
    p1, q1 = _layer1_predict(X1, assets['layer1'], assets['layer1_calibrator'])
    p1, q1 = p1[0], q1[0]
    classes = [int(c) + 1 for c in assets['layer1_calibrator']['classes']]
    layer1 = {'classes': classes, 'proba': [float(v) for v in p1],
              'score_raw': [float(v) for v in q1],
              'recommend': classes[int(np.argmax(q1))],   # v1.1：未校准 argmax
              'recommend_basis': 'uncalibrated_argmax'}

    layer2 = {}
    for t in BIN_TARGETS:
        Xt = build_feature_frame([raw], fl[t], med[t])
        p = float(assets['layer2'][t]['model'].predict_proba(Xt.values)[0, 1])
        layer2[t] = {'proba': p, 'threshold': thr[t], 'predict_effective': bool(p >= thr[t])}

    return {'layer1': layer1, 'layer2': layer2}


def predict_batch(assets: dict, raw_df: pd.DataFrame) -> pd.DataFrame:
    """批量推理。raw_df 只需包含用到的原始变量列，缺列自动按缺失处理。"""
    fl, med, thr = assets['feature_lists'], assets['medians'], assets['thresholds']
    raws = raw_df.to_dict('records')
    out = pd.DataFrame(index=raw_df.index)

    X1 = _layer1_frame(raws, fl[CLS_TARGET])
    p1, q1 = _layer1_predict(X1, assets['layer1'], assets['layer1_calibrator'])
    classes = [int(c) + 1 for c in assets['layer1_calibrator']['classes']]
    for j, c in enumerate(classes):
        out[f'layer1_proba_方案{c}'] = p1[:, j]        # 校准后概率（把握度参考）
        out[f'layer1_原始得分_方案{c}'] = q1[:, j]      # 未校准得分（推荐依据）
    out['layer1_推荐方案'] = [classes[i] for i in np.argmax(q1, axis=1)]  # v1.1：未校准 argmax

    for t in BIN_TARGETS:
        Xt = build_feature_frame(raws, fl[t], med[t])
        p = assets['layer2'][t]['model'].predict_proba(Xt.values)[:, 1]
        out[f'{t}_有效概率'] = p
        out[f'{t}_判定'] = np.where(p >= thr[t], '有效', '无效')
    return out
