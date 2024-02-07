# 必要なライブラリのインポート
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd

# 特徴抽出関数
def extract_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        return np.array(fp)
    else:
        return None

# モデル定義関数
def create_model(input_shape):
    model = Sequential()
    model.add(Dense(512, input_dim=input_shape, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='linear'))  # 予測対象が1つの出力値
    model.compile(optimizer='adam', loss='mse')
    return model

# データの読み込みと前処理
def load_and_preprocess_data(filepath):
    dataframe = pd.read_csv(filepath)
    features = np.array([extract_features(smiles) for smiles in dataframe['SMILES']])
    targets = dataframe['IC50'].values
    return features, targets

# モデルのトレーニングと評価
def train_and_evaluate_model(features, targets):
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)
    model = create_model(input_shape=2048)
    model.fit(X_train, y_train, epochs=100, validation_split=0.2)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse}")
    return model

# メイン関数
if __name__ == "__main__":
    filepath = 'path_to_your_data.csv'
    features, targets = load_and_preprocess_data(filepath)
    model = train_and_evaluate_model(features, targets)
    # モデルを保存
    model.save('my_model.h5')
