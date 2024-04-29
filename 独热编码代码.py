import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef, roc_curve, auc
import tensorflow as tf

# 加载数据集
df = pd.read_excel(r'D:\download\pycharm\深度学习 抗菌肽\4000个被证实.xlsx')
sequences = df['Sequence'].astype(str).values  # 序列数据
labels = df['Label'].values  # 标签数据

# 使用Keras的Tokenizer进行序列的编码
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(sequences)
sequences_encoded = tokenizer.texts_to_sequences(sequences)
max_len = 100  # 设定序列的最大长度
sequences_padded = pad_sequences(sequences_encoded, maxlen=max_len)
sequences_one_hot = to_categorical(sequences_padded)
vocab_size = len(tokenizer.word_index) + 1  # 获取词汇表大小

# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(sequences_one_hot, labels, test_size=0.2, random_state=42)

# 构建序列模型
model = Sequential([
    Conv1D(64, 3, activation='relu', input_shape=(max_len, vocab_size)),
    MaxPooling1D(3),
    LSTM(64),
    Flatten(),
    Dense(100, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(X_train, y_train, epochs=30, batch_size=25, validation_data=(X_test, y_test))

# 保存模型和Tokenizer
model_path = r'D:\毕业论文\抗菌肽\抗菌肽模型\独热4000个被证实.keras'
model.save(model_path)
tokenizer_path = r'D:\毕业论文\抗菌肽\抗菌肽模型\独热tokenizer.pickle'
with open(tokenizer_path, 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# 使用模型进行预测
predictions_probs = model.predict(X_test)

# 计算并打印评估指标
accuracy = accuracy_score(y_test, (predictions_probs > 0.5).astype(int))
tn, fp, fn, tp = confusion_matrix(y_test, (predictions_probs > 0.5).astype(int)).ravel()
sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
mcc = matthews_corrcoef(y_test, (predictions_probs > 0.5).astype(int))

# 计算ROC曲线
fpr, tpr, thresholds = roc_curve(y_test, predictions_probs.ravel())
roc_auc = auc(fpr, tpr)

# 绘制ROC曲线
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# 指标数据
metrics = ['Accuracy', 'Sensitivity', 'Specificity', 'MCC']
values = [accuracy, sensitivity, specificity, mcc]

# 创建条形图
plt.figure(figsize=(10, 5))
bars = plt.bar(metrics, values, color=['blue', 'green', 'red', 'purple'])
# 在每个条上添加数值
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')

plt.xlabel('Metrics')
plt.ylabel('Values')
plt.title('Model Performance Metrics')
plt.ylim([0, 1])  # 设置y轴的范围从0到1
plt.axhline(y=1, color='gray', linestyle='--')  # 画一条y=1的虚线作为参考
plt.show()

# 输出结果
print(f'Accuracy: {accuracy:.2f}')
print(f'Sensitivity: {sensitivity:.2f}')
print(f'Specificity: {specificity:.2f}')
print(f'MCC: {mcc:.2f}')
print(f'ROC AUC: {roc_auc:.2f}')

