# predict breast cancer in Python
[predict web page](https://breast.predict.nhs.uk/) / [predict git hub](https://github.com/WintonCentre/predict-v30-r)
## method
参照：https://breast.predict.nhs.uk/predict-mathematics.pdf  

乳がんが原因のベースライン累積ハザード関数   

ER+のとき  
$$H_{cancer}(t) = exp[\ 0.7424402 − 7.527762/\sqrt{t} − 1.812513 ∗ log(t)/\sqrt{t} \ ]$$
ER-のとき
$$H_{cancer}(t) = exp[\ −1.156036 + 0.4707332/t^2 − 3.51355/t \ ]$$

その他が原因のベースライン累積ハザード関数  
$$H_{other}(t) = exp[\ −6.052919 + (1.079863 * log(t)) + (0.3255321 * \sqrt{t}) \ ]$$


---
## 使い方

### install package
```sh
pip install scikit-survival
```
---

## input dataについて
|カラム名|説明|
|----|----|
|pID|症例番号|
|age|患者の年齢|
|tumour_size|腫瘍の大きさ|
|tumour_grade|腫瘍のグレード|
|nodes|リンパ節転移の数|
|er|ER status|
|her2|HER2 status|
|ki67|Ki-67 status|
|chemo_generation|Chemotherapy generation|
|hormonetherapy|Hormone (endocrine) therapy|
|trastuzumab|Trastuzumab|
|bisphosphonates|Bisphosphonates|
|event|event発生の有無|
|duration|event発生までの時間|


## Patientクラスについて

Patinet クラスは患者一人に対してpredictの生存時間解析を行うクラスです。
入力は(1, 15)のshapeのデータフレームを用います。
データフレームのカラムは
pID, age, detection, tumour_size, tumour_grade, nodes, er,her2ki67, chemo_generation, hormonetherapy,trastuzumab, bisphosphonates, event, duration の15列です。

```
    """Patient class

    Patientごとの生存時間解析を行うクラス
    
    Attributes:
        # 患者の情報
        age (int): 患者の年齢
        detection (int): 発見契機 symptoms detected = 0, screen detected = 1, unknown = 2
        size (float): Tumour size (mm)
        grade (int) : Tumour grade 1, 2 or 3 only
        nodes (int) : がんが転移したリンパ節の数
        er (int) : ER+ = 1, ER- = 0
        her2 (int) : her2+ = 1, her2- = 0 missing = 2
        ki67 (int) : ki67+ = 1, ki67- = 0 missing = 2
        event (int) : event発生の有無 0, 1
        duration (int) : event発生時間
        
        # Tretment Options
        chemo_gen (int) : Chemo generation 0, 2 or 3 only 
        horm (int) :  Hormone therapy Yes = 1, no = 0
        traz (int) : Trastuzumab therapy Yes = 1, no = 0
        bis (int) : Bisphosphonate therapy Yes = 1, no = 0
        treatment_options (str) : 患者が行っている治療法の結合 -> "c0:h1:t0:b1"
        
        # other
        max_time (int) : 打ち切りまでの時間（年）
        time 
        coef_ : 回帰係数
        _VERSION : predictのバージョン
    
    """
```

###  get_patient_data メソッド
入力情報を辞書式で返すメソッド

```python
from predict import Patient

# データのロード
input_data = pd.read_csv('./data/input_data3.csv')
patient_data = input_data.query(" pID==1 ")

#インスタンス化
patient = Patient(patient_data=patient_data)
patient.get_patient_data()
```
```python
>>>{'pID': 1,
    'age': 26,
    'detection': 0,
    'tumour_size': 2,
    'tumour_grade': 2,
    'nodes': 9,
    'er': 1,
    'her2': 1,
    'ki67': 0,
    'chemo_generation': 0,
    'hormonetherapy': 0,
    'trastuzumab': 0,
    'bisphosphonates': 0,
    'event': 1,
    'duration': 13}
```

###  predict メソッド
risk_scoreを返すメソッド
```python
patient.predict()
>>> 1.8450373070201214
```

### predict_cumulative_hazard_function メソッド
累積ハザード関数を返すメソッド
```python
patient.predict_cumulative_hazard_function()
>>> array([ 0.76015913,  2.74531324,  5.49845931,  8.67575268, 12.06248254,
       15.52440702, 18.97699276, 22.36709505, 25.66182112, 28.84156452,
       31.89553415, 34.8188219 , 37.61044525, 40.27202257, 42.80686716])
```

### predict_survival_function メソッド
生存関数を返すメソッド
```python
patient.predict_survival_function()
>>> array([99.23984087, 97.25468676, 94.50154069, 91.32424732, 87.93751746,
       84.47559298, 81.02300724, 77.63290495, 74.33817888, 71.15843548,
       68.10446585, 65.1811781 , 62.38955475, 59.72797743, 57.19313284])
```

## concordance_score 関数について
concordance_scoreスコアを返す関数
sksurvのconcordance_index_censoredを使用しています。

```python
from predict import concordance_score

input_data = pd.read_csv('./data/input_data3.csv')
concordance_score(input_data)

>>> 0.5944954128440367
```