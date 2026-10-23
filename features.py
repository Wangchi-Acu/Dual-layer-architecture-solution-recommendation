# -*- coding: utf-8 -*-
"""
features.py — 确定性特征工程（与训练管线同逻辑）

规则（已在真实数据上逐列验证）：
    1) 分段标志：  'X_geN' = (X >= N)，如 BPSQI_ge14 = (BPSQI >= 14)
                   支持 _ge/_gt/_le/_lt 后缀；
    2) 交互项：    'A*B' = A × B，其中 A、B 可以是原始变量，也可以是分段标志
                   （如 SEX*BPSQI_ge14），递归解析；
    3) 原始变量：  直接使用录入值。

任何无法解析的取值（缺失/非数值）返回 NaN，由 inference 层用训练集中位数填补。
正式替换模型时：若训练管线的 engineer_features 规则有变化，只需同步修改本文件。
"""
import re
import math

SEG_PATTERN = re.compile(r'^(.+)_(ge|le|gt|lt)(\d+(?:\.\d+)?)$')


def _to_float(v):
    if v is None:
        return float('nan')
    try:
        f = float(v)
        return f if math.isfinite(f) else float('nan')
    except (TypeError, ValueError):
        return float('nan')


def resolve_feature(name: str, raw: dict):
    """按名称递归解析单个特征。raw: {原始变量名: 数值或None}。"""
    if name in raw:
        return _to_float(raw[name])

    if '*' in name:
        a, b = name.split('*', 1)
        va, vb = resolve_feature(a, raw), resolve_feature(b, raw)
        if math.isnan(va) or math.isnan(vb):
            return float('nan')
        return va * vb

    m = SEG_PATTERN.match(name)
    if m:
        base, op, thr_s = m.group(1), m.group(2), m.group(3)
        v = resolve_feature(base, raw)
        if math.isnan(v):
            return float('nan')
        thr = float(thr_s)
        if op == 'ge':
            return float(v >= thr)
        if op == 'gt':
            return float(v > thr)
        if op == 'le':
            return float(v <= thr)
        return float(v < thr)

    # 未知的原始变量（表单未提供该字段）
    return _to_float(raw.get(name))


def raw_vars_for(feature_lists: dict) -> set:
    """从若干特征清单反推需要录入的原始变量集合。"""
    out = set()

    def walk(name):
        if '*' in name:
            a, b = name.split('*', 1)
            walk(a)
            walk(b)
            return
        m = SEG_PATTERN.match(name)
        if m:
            walk(m.group(1))
            return
        out.add(name)

    for feats in feature_lists.values():
        for f in feats:
            walk(f)
    return out
