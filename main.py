import japanize_matplotlib

# データ前処理


import numpy as np
import pandas as pd

# データ可視化
from matplotlib import rcParams
import matplotlib.pyplot as plt

# import matplotlib.pyplot 
import seaborn as sns
plt.style.use("ggplot")
# %matplotlib inline

# グラフの日本語表記対応

rcParams["font.family"]     = "sans-serif"
rcParams["font.sans-serif"] = "Hiragino Maru Gothic Pro"

# データセット読込
from sklearn.datasets import load_boston
boston = load_boston()

# DataFrame作成
df = pd.DataFrame(boston.data)
df.columns = boston.feature_names
df["MEDV"] = boston.target


# 相関係数を算出
corr = np.corrcoef(df.values.T)

# # ヒートマップとして可視化
# hm   = sns.heatmap(
#                  corr,                         # データ
#                  annot=True,                   # セルに値入力
#                  fmt='.2f',                    # 出力フォーマット
#                  annot_kws={'size': 6},        # セル入力値のサイズ
#                  yticklabels=list(df.columns), # 列名を出力
#                  xticklabels=list(df.columns)) # x軸を出力

# plt.tight_layout()
# plt.show()


# # 目的変数（MEDV)と相関が強い上位5つの変数を散布図として可視化
# sns.pairplot(df[['INDUS', 'RM', 'TAX', 'PTRATIO','LSTAT', 'MEDV']])
# plt.tight_layout()
# plt.show()

# # 目的変数(MEDV)と相関が強い変数としてRM（部屋数）を選択し可視化
# sns.lmplot("RM","MEDV",data=df)
# plt.tight_layout()
# plt.show()

# # 目的変数(MEDV)のヒストグラムを表示
# plt.hist(df.MEDV,bins=60, density=True)
# plt.xlabel("MEDV(住宅価格の中央値)", fontname ='MS Gothic')
# plt.ylabel("住宅数", fontname ='MS Gothic')
# plt.tight_layout()
# plt.show()


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 変数定義
X = df[['RM']].values # 説明変数
y = df['MEDV'].values # 目的変数

# 学習・テストデータ分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# 回帰モデルのインスタンス
model = LinearRegression(
                         fit_intercept=True, # 切片の計算
                         normalize=True,    # 正規化
                         copy_X=True,        # メモリ内でのデータ複製
                         n_jobs=1,           # 計算に用いるジョブの数
                         positive=False
                        )

# モデル学習
model.fit(X_train, y_train)

# 予測値(Train）
y_train_pred = model.predict(X_train)

# 予測値（Test)


y_test_pred = model.predict(X_test)
y_test_pred_value = model.predict([[5.6]])
print("y_test_pred_value",y_test_pred_value)

# # 回帰モデルのインスタンス
# model = LinearRegression(
#                          fit_intercept=True,
#                          normalize=False,
#                          copy_X=True, 
#                          n_jobs=1,
#                          positive=True
#                         )


# 単回帰直線を可視化
plt.scatter(X_train, y_train, c='blue', s=30)       # 散布図
plt.plot(X_train, y_train_pred, color='red', lw=4)  # 直線

# グラフの書式設定
plt.xlabel('RM(平均部屋数/一戸)', fontname ='MS Gothic')
plt.ylabel('MEDV(住宅価格の中央値)', fontname ='MS Gothic')
plt.tight_layout()
plt.show()

# 回帰係数情報出力
print('偏回帰係数: %.2f' % model.coef_[0])
print('切片: %.2f' % model.intercept_)

# 出力結果
# 偏回帰係数: 9.31
# 切片: -35.99

# 予測値と残差をプロット（学習データ）
plt.scatter(y_train_pred,             # グラフのx値(予測値)  
            y_train_pred - y_train,   # グラフのy値(予測値と学習値の差)
            c='blue',                 # プロットの色
            marker='o',               # マーカーの種類
            s=40,                     # マーカーサイズ
            alpha=0.7,                # 透明度
            label='学習データ')         # ラベルの文字


# 予測値と残差をプロット（テストデータ）
plt.scatter(y_test_pred,            
            y_test_pred - y_test, 
            c='red',
            marker='o', 
            s=40,
            alpha=0.7,
            label='テストデータ')

# グラフの書式設定
plt.xlabel('予測値', fontname ='MS Gothic')
plt.ylabel('残差', fontname ='MS Gothic')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-20, xmax=60, lw=2, color='black')
plt.xlim([-10, 50])
plt.ylim([-40, 20])
plt.tight_layout()
plt.show()

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 変数定義
X = df[['INDUS','RM','TAX','PTRATIO','LSTAT']].values # 説明変数
y = df['MEDV'].values                                 # 目的変数（住宅価格の中央値）

# 学習データ/テストデータ分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# 重回帰のインスタンス
model_multi = LinearRegression()

# モデル学習
model_multi.fit(X_train, y_train)

# 予測値(学習データ）
y_train_pred = model_multi.predict(X_train)

# 予測値（テストデータ)
y_test_pred  = model_multi.predict(X_test)


# 偏回帰係数
df_coef = pd.DataFrame(model_multi.coef_.reshape(1,5), 
                       columns=['INDUS','RM','TAX','PTRATIO','LSTAT'])
print(df_coef)

# 切片
print('切片: %.2f' % model_multi.intercept_)


# 出力結果（偏回帰係数）
#      INDUS        RM       TAX   PTRATIO     LSTAT
#   0.056633  4.627714 -0.005232 -1.017411 -0.526502

# 出力結果（切片）
# 20.37

from sklearn.metrics import r2_score            # 決定係数
from sklearn.metrics import mean_squared_error  # RMSE

# 平均平方二乗誤差(RMSE)
print('RMSE 学習: %.2f, テスト: %.2f' % (
        mean_squared_error(y_train, y_train_pred, squared=False), # 学習
        mean_squared_error(y_test, y_test_pred, squared=False)    # テスト
      ))

# 決定係数(R^2)
print('R^2 学習: %.2f, テスト: %.2f' % (
        r2_score(y_train, y_train_pred), # 学習
        r2_score(y_test, y_test_pred)    # テスト
      ))

# 出力結果
# RMSE 学習: 4.92, テスト: 5.83
# R^2  学習: 0.71, テスト: 0.59

# 予測値と残差をプロット（学習データ）
plt.scatter(y_train_pred,             # グラフのx値(予測値)  
            y_train_pred - y_train,   # グラフのy値(予測値と学習値の差)
            c='blue',                 # プロットの色
            marker='o',               # マーカーの種類
            s=40,                     # マーカーサイズ
            alpha=0.7,                # 透明度
            label='学習データ')         # ラベルの文字


# 予測値と残差をプロット（テストデータ）
plt.scatter(y_test_pred,            
            y_test_pred - y_test, 
            c='red',
            marker='o', 
            s=40,
            alpha=0.7,
            label='テストデータ')

# グラフ書式設定
plt.xlabel('予測値', fontname ='MS Gothic')
plt.ylabel('残差', fontname ='MS Gothic')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-20, xmax=60, lw=2, color='black')
plt.xlim([-20, 60])
plt.ylim([-40, 20])
plt.tight_layout()
plt.show()