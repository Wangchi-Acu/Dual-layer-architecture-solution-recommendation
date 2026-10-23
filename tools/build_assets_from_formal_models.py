# -*- coding: utf-8 -*-
"""
build_assets_from_formal_models.py — 由训练管线定稿产物重建应用资产（provenance 记录）

本应用的 models/ 资产来自：
  第一层（三分类）：三分类管线 05_最终模型_backward_tuned/SVM.joblib
                   + calibration/calibrator_sigmoid.joblib
  第二层（二分类）：M5 v8 消融驱动全流程输出 y_binN/models/final_model_y_binN.pkl

本脚本记录从上述产物重建 feature_lists.json / train_medians.json / thresholds.json
以及 tests/fixtures/expected_predictions.json 的过程，供审计与再生成使用。

用法：
    python tools/build_assets_from_formal_models.py --layer1-dir /path/05_最终模型_backward_tuned \
        --layer2-dir /path/二分类输出根目录（含 y_bin1/2/3 子目录）
"""
import argparse
import json
import os
import sys

import joblib

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN_TARGETS = ['y_bin1', 'y_bin2', 'y_bin3']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--layer1-dir', required=True, help='含 SVM.joblib 与 calibration/ 的目录')
    ap.add_argument('--layer2-dir', required=True, help='含 y_bin1/2/3/models/final_model_*.pkl 的根目录')
    args = ap.parse_args()

    mdir = os.path.join(PROJECT_ROOT, 'models')
    os.makedirs(mdir, exist_ok=True)

    # 第一层
    l1_files = [f for f in os.listdir(args.layer1_dir) if f.endswith('.joblib')]
    assert len(l1_files) == 1, f'第一层目录应有且仅有一个 joblib：{l1_files}'
    b1 = joblib.load(os.path.join(args.layer1_dir, l1_files[0]))
    cal = joblib.load(os.path.join(args.layer1_dir, 'calibration', 'calibrator_sigmoid.joblib'))
    joblib.dump(b1, os.path.join(mdir, 'layer1_multiclass.pkl'))
    joblib.dump(cal, os.path.join(mdir, 'layer1_calibrator.pkl'))

    # 第二层
    bundles = {}
    for t in BIN_TARGETS:
        src = os.path.join(args.layer2_dir, t, 'models', f'final_model_{t}.pkl')
        bundles[t] = joblib.load(src)
        joblib.dump(bundles[t], os.path.join(mdir, f'layer2_{t}.pkl'))

    feature_lists = {'y_cls': list(b1['features']),
                     **{t: list(bundles[t]['features']) for t in BIN_TARGETS}}
    medians = {'y_cls': {},
               **{t: {k: float(v) for k, v in bundles[t]['fill_median'].items()} for t in BIN_TARGETS}}
    thresholds = {t: float(bundles[t]['youden_threshold']) for t in BIN_TARGETS}

    json.dump(feature_lists, open(os.path.join(mdir, 'feature_lists.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    json.dump(medians, open(os.path.join(mdir, 'train_medians.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    json.dump(thresholds, open(os.path.join(mdir, 'thresholds.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print('模型与 JSON 已重建。fixtures 的再生成需在第一层管线环境（含原版 common.py）下进行，'
          '参见 tests/fixtures/expected_predictions.json 头部注释与研究管线 07_predict_calibrated.py。')


if __name__ == '__main__':
    main()
