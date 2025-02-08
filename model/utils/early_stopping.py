import torch
import numpy as np

class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=False):
        """
        Args:
            patience (int): 개선되지 않는 에포크 수를 기준으로 학습 중단
            delta (float): 성능 개선으로 간주할 최소 변화량
            path (str): 모델 가중치를 저장할 경로
            verbose (bool): 개선 시 출력 여부
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score

        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

