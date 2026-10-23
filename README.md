# 失眠针刺方案智能推荐系统（双层架构）

基于多模态基线特征的失眠症针刺方案双层智能推荐模型网页应用。

- **第一层（方案推荐）**：RBF 支持向量机（宽度学习实现）三分类模型，15 个原始输入特征
  （人口学 2 项、问卷/条目 5 项、HRV 与熵指标 8 项）。**v1.1 起推荐类别取未校准 softmax 得分的 argmax**
  （与训练管线 OOF 评估口径一致）；OvR sigmoid（Platt）校准后概率仍输出，仅供把握度参考、不参与推荐
  （校准口径因决策值动态范围过窄退化为恒定推荐方案 3，已弃用为推荐依据）。
  冻结测试集（n=45）：准确率 0.800、宏平均 F1 0.785、OvR 宏平均 AUC 0.788。
- **第二层（推荐方案疗效预测）**：三个二分类模型分别预测各方案"显效"概率，
  全部由两两交互特征构成，经消融驱动"算法×特征数"联合寻优定稿：
  - 方案一（通督调卫）：极端随机树 @ 3 特征（F1@0.5 = 0.854±0.010，OOF AUC 0.725；测试集 AUC 0.867）
  - 方案二（通督调脏）：梯度提升 @ 8 特征（F1@0.5 = 0.800±0.018，OOF AUC 0.782；测试集 AUC 0.827）
  - 方案三（通督养心）：极端随机树 @ 11 特征（F1@0.5 = 0.867±0.012，OOF AUC 0.681；测试集 AUC 0.898）
  - 判定阈值取各自训练集 OOF 的 Youden 最优值（0.552 / 0.738 / 0.493）
  - **v1.2 起**：第一层完成方案推荐后，界面只展示该推荐方案的显效概率与阈值判定，
    并同步展示对应方案的穴位定位示意图（动图，assets/acupoint/）；
    不再展示三方案并列比较与两层一致性提示（三个二分类模型各自独立定标，概率不可跨方案直接比较）。
  - **v1.3**：按编码本更正方案命名（1=通督调卫、2=通督调脏、3=通督养心，模型与数值不变）；
    录入变量标签按编码本补全（SF=主诉、DI=合并疾病史、SS=伴随症状、PA–PG=PSQI 成分）；
    SEX/EDU/MAR/OCC 改为带含义下拉框。穴位图文件名已同步更正，
    图内标题仍为旧编号，将由研究者自有版本同名覆盖替换。

> 本系统仅供科研参考，不构成临床诊断或治疗依据。

## 目录结构

```
acupuncture_app/
├── app.py                  # Streamlit 入口
├── inference.py            # 资产加载与双层推理（不依赖 streamlit，可独立测试）
├── features.py             # 第二层特征工程：A*B 交互项、X_geN 分段标志递归解析
├── common.py               # 第一层模型运行时依赖（BroadClassifier / transform / score_matrix，
│                           #   与研究管线同名模块，pkl 反序列化必需，勿改名）
├── config.yaml             # 应用配置与版本号
├── requirements.txt        # 依赖（scikit-learn 锁定 1.7.2，与训练环境一致）
├── models/
│   ├── layer1_multiclass.pkl       # 第一层模型包（含预处理状态）
│   ├── layer1_calibrator.pkl       # 第一层 OvR Platt 校准器
│   ├── layer2_y_bin1/2/3.pkl       # 第二层自包含模型包（模型+特征+中位数+阈值）
│   ├── feature_lists.json          # 各层特征清单（与 pkl 内一致，供自检）
│   ├── train_medians.json          # 第二层训练集中位数填补值
│   └── thresholds.json             # 第二层 Youden 阈值
├── assets/
│   ├── var_meta.csv        # 录入变量元数据（分组/标签/范围/默认值/分类编码）
│   ├── example_patients.csv# 录入模板（3 行真实示例患者，三类各一）
│   └── acupoint/           # 三方案穴位定位示意图（静态 png + 动图 gif，config.yaml 可配）
├── tests/
│   ├── test_consistency.py         # 数值一致性回归测试（容差 1e-6）
│   └── fixtures/expected_predictions.json
└── tools/
    └── build_assets_from_formal_models.py  # 从训练产物重建 models/ 资产（provenance）
```

## 本地运行

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 部署（Streamlit Community Cloud）

本项目已完成在线部署：

- 代码仓库（GitHub）：https://github.com/Wangchi-Acu/Dual-layer-architecture-solution-recommendation
- 在线访问地址：https://dual-layer-architecture-solution-recommendation.streamlit.app/

自行复刻部署的流程：

1. 将本目录全部文件推送到 GitHub 仓库；
2. share.streamlit.io → New app → 选择仓库、分支与 `app.py`；
3. 启动时应用会自动进行自检（sklearn 版本、特征清单一致性、预测接口），失败会阻止启动并显示原因。

## 模型替换 / 再生成

models/ 下的 pkl 与三个 JSON 必须来自同一次训练定稿，切勿混用不同批次的文件。
从训练管线产物重建资产：

```bash
python tools/build_assets_from_formal_models.py \
    --layer1-dir /path/05_最终模型_backward_tuned \
    --layer2-dir /path/二分类输出根目录   # 含 y_bin1/2/3 子目录
```

替换后必须运行一致性回归测试：

```bash
python tests/test_consistency.py
```

## 说明

- 录入端仅需 **41 个原始变量**（见 assets/var_meta.csv）；第二层所需的交互项与分段标志由
  features.py 在线展开，无需手工计算。
- SEX（性别）、EDU（受教育程度）、MAR（婚姻状况）、OCC（职业）为整型编码的分类变量，
  单例录入为带含义下拉选择（如 0=小学及以下、1=男）；批量上传 CSV 时填编码值即可。
  编码含义见 var_meta.csv 的 option_labels 列及《基线特征变量_中英双语映射表》。
- 第一层同时输出未校准 softmax 得分（推荐依据）与 Platt 校准后概率（把握度参考）。
