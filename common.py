# -*- coding: utf-8 -*-
"""
common.py — 三分类第一层模型（RBF SVM / BroadClassifier）运行时所依赖的最小模块。

第一层模型文件 models/layer1_multiclass.pkl 由研究管线（04_代码/common.py）训练并
序列化，pickle 按模块名 `common` 引用 BroadClassifier 类，因此本文件必须以
`common.py` 为名随应用一起部署，且类名/方法签名不得修改。

本文件只保留推理所需内容（BroadClassifier、transform、score_matrix），
训练期依赖（random_numbers、partition、梯度提升三库等）不包含在内。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from scipy.special import softmax
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC

SEED = 919


class BroadClassifier(ClassifierMixin, BaseEstimator):
    """宽度学习风格分类器：RBF 核随机特征映射 + 线性头部（与研究管线一致）。"""

    def __init__(self, objective='logistic', scale=10.0, C=1.0, alpha=1e-4, eta0=0.01,
                 fit_intercept=True):
        self.objective = objective
        self.scale = scale
        self.C = C
        self.alpha = alpha
        self.eta0 = eta0
        self.fit_intercept = fit_intercept

    def _kernel(self, X, Y):
        return np.exp(-self.gamma_ * cdist(X, Y, 'sqeuclidean'))

    def _features(self, X):
        return self._kernel(X, self.X_fit_) @ self.map_

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self.X_fit_ = np.asarray(X, float).copy()
        d = cdist(self.X_fit_, self.X_fit_, 'sqeuclidean')
        v = d[np.triu_indices(len(self.X_fit_), 1)]
        med = np.median(v[v > 0])
        self.gamma_ = float(self.scale / med)
        K = self._kernel(self.X_fit_, self.X_fit_)
        val, vec = eigh(K, check_finite=False)
        keep = val > max(1e-12, val[-1] * 1e-12)
        self.map_ = vec[:, keep] / np.sqrt(val[keep])
        Z = vec[:, keep] * np.sqrt(val[keep])
        if self.objective == 'svm_squared':
            self.estimator_ = LinearSVC(C=self.C, fit_intercept=self.fit_intercept,
                                        loss='squared_hinge', dual='auto', max_iter=30000,
                                        tol=1e-6, random_state=SEED)
        elif self.objective == 'perceptron':
            self.estimator_ = SGDClassifier(loss='perceptron', alpha=self.alpha,
                                            fit_intercept=self.fit_intercept,
                                            learning_rate='constant', eta0=self.eta0,
                                            penalty='l2', max_iter=3000, tol=1e-5,
                                            n_iter_no_change=20, random_state=SEED)
        elif self.objective == 'logistic':
            self.estimator_ = LogisticRegression(C=self.C, fit_intercept=self.fit_intercept,
                                                 max_iter=3000, tol=1e-7, random_state=SEED,
                                                 solver='lbfgs')
        else:
            raise ValueError(self.objective)
        self.estimator_.fit(Z, y)
        return self

    def predict(self, X):
        return np.asarray(self.estimator_.predict(self._features(X))).reshape(-1)

    def decision_function(self, X):
        Z = self._features(X)
        if hasattr(self.estimator_, 'predict_proba'):
            return self.estimator_.predict_proba(Z)
        return self.estimator_.decision_function(Z)

    def predict_proba(self, X):
        Z = self._features(X)
        if hasattr(self.estimator_, 'predict_proba'):
            return self.estimator_.predict_proba(Z)
        s = np.asarray(self.estimator_.decision_function(Z), float)
        return softmax(s if s.ndim > 1 else np.c_[-s, s], axis=1)


def transform(frame: pd.DataFrame, state):
    """按训练时保存的预处理状态（中位数填补 + 标准化 / 众数填补 + 独热）变换输入。"""
    parts = []
    if state['numeric'] is not None:
        cols, im, sc = state['numeric']
        parts.append(sc.transform(im.transform(frame[cols])))
    if state['categorical'] is not None:
        cols, im, enc = state['categorical']
        parts.append(enc.transform(im.transform(frame[cols])))
    return np.concatenate(parts, 1)


def score_matrix(model, X):
    """输出 3 列得分矩阵（优先 predict_proba，否则 decision_function 的 softmax）。"""
    if hasattr(model, 'predict_proba'):
        q = np.asarray(model.predict_proba(X), float)
    elif hasattr(model, 'decision_function'):
        s = np.asarray(model.decision_function(X), float)
        q = softmax(s if s.ndim > 1 else np.c_[-s, s], axis=1)
    else:
        pred = model.predict(X)
        q = np.eye(3)[pred]
    if q.shape[1] != 3:
        z = np.zeros((len(q), 3))
        for j, c in enumerate(model.classes_):
            z[:, int(c)] = q[:, j]
        q = z
    return q
