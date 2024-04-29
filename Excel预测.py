import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
import pickle

def load_model_and_tokenizer(model_path, tokenizer_path):
    # 加载模型和 tokenizer
    model = load_model(model_path)
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

def predict_file(input_path, output_path, model, tokenizer):
    # 从 Excel 文件中读取数据
    df = pd.read_excel(input_path)
    sequences = df['Sequence'].astype(str).values
    sequences_encoded = tokenizer.texts_to_sequences(sequences)
    max_len = 100  # 根据模型的输入要求设置最大长度
    sequences_padded = pad_sequences(sequences_encoded, maxlen=max_len)
    # 使用独热编码转换，vocab_size 应设置为tokenizer中的词汇量加1
    vocab_size = len(tokenizer.word_index) + 1
    sequences_one_hot = to_categorical(sequences_padded, num_classes=vocab_size)
    predictions = model.predict(sequences_one_hot)
    df['Prediction'] = predictions.ravel()  # 将预测结果添加到 DataFrame

    # 将结果保存到新的 Excel 文件中
    df.to_excel(output_path, index=False)
    print("预测结果已成功保存至 " + output_path)

# 指定模型和 tokenizer 的路径
model_path = r'D:\毕业论文\抗菌肽\抗菌肽模型\独热4000个被证实.keras'
tokenizer_path = r'D:\毕业论文\抗菌肽\抗菌肽模型\独热tokenizer.pickle'

# 加载模型和 tokenizer
model, tokenizer = load_model_and_tokenizer(model_path, tokenizer_path)

# 指定输入和输出文件的路径
input_path = r'D:\download\pycharm\深度学习 抗菌肽\5000个被证实.xlsx'
output_path = r'D:\download\pycharm\深度学习 抗菌肽\5000个被证实_predictions.xlsx'

# 进行预测并保存结果
predict_file(input_path, output_path, model, tokenizer)
