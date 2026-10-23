# -*- coding: utf-8 -*-
"""
test_consistency.py — 模型替换后的数值一致性回归测试

用法：python tests/test_consistency.py
原理：用固定示例患者的原始变量走完整推理路径（第一层 transform+校准 /
     第二层交互特征工程+中位数填补→predict_proba），
     与 fixtures/expected_predictions.json 中的期望输出逐值比对（容差 1e-6）。
fixtures 由 tools/build_assets_from_formal_models.py 在训练管线口径下生成。
"""
import os
import sys
import json

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from inference import load_assets, predict_one, BIN_TARGETS  # noqa: E402

TOL = 1e-6


def main():
    assets = load_assets(PROJECT_ROOT)
    with open(os.path.join(PROJECT_ROOT, 'tests', 'fixtures', 'expected_predictions.json'),
              encoding='utf-8') as f:
        fixtures = json.load(f)

    n_fail = 0
    recs = []
    for case_id, case in fixtures.items():
        got = predict_one(assets, case['raw'])
        expect = case['expect']
        recs.append(got['layer1']['recommend'])

        ok = np.allclose(got['layer1']['proba'], expect['layer1_proba'], atol=TOL)
        print(f"{case_id} layer1     : {'PASS' if ok else 'FAIL'}  {np.round(got['layer1']['proba'], 6)}")
        n_fail += 0 if ok else 1

        # v1.1 哨兵：推荐须取未校准 argmax（校准口径会退化为恒定方案3）
        ok = got['layer1']['recommend'] == got['layer1']['classes'][int(np.argmax(got['layer1']['score_raw']))]
        print(f"{case_id} 推荐口径   : {'PASS' if ok else 'FAIL'}  recommend={got['layer1']['recommend']}")
        n_fail += 0 if ok else 1

        for t in BIN_TARGETS:
            g, e = got['layer2'][t]['proba'], expect[f'{t}_proba']
            ok = abs(g - e) <= TOL
            print(f"{case_id} {t:8s}: {'PASS' if ok else 'FAIL'}  got={g:.8f} expect={e:.8f}")
            n_fail += 0 if ok else 1

    print()
    # v1.1 哨兵：三名示例患者的推荐不得恒定（恒定 = 校准退化 bug 复发）
    ok = len(set(recs)) >= 2
    print(f"推荐多样性哨兵: {'PASS' if ok else 'FAIL'}  {recs}")
    n_fail += 0 if ok else 1
    print()
    if n_fail == 0:
        print('全部通过：网页推理路径与期望输出一致。')
        sys.exit(0)
    print(f'{n_fail} 项不一致——请检查 pkl / JSON / common.py 是否配套。')
    sys.exit(1)


if __name__ == '__main__':
    main()
