# -LSTM-
在双向 LSTM 上引入自注意力的 Attn‑BiLSTM，并构建卷积—时序双通道并行融合注意力模型（Co‑Attn），以增强特征表征与证据聚合的准确性与可解释性。在 IMDB 影评数据集（https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)（50000 条样本，训练/测试=80/20）上进行评测，统一词表规模 20000 与序列长度 200，采用嵌入维度 128、LSTM 隐层 64、批量 64、epoch=5、Adam(lr=0.001) 与 BCEWithLogitsLoss，并在 CPU 环境完成训练与验证。
