import pandas as pd
from flask import Flask, render_template, url_for, request
import pymysql
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chart')
def chart():
    date = []
    a = []
    _sum1 = 0
    _sum2 = 0
    _sum3 = 0
    _sum4 = 0
    try:
        sample_db = pymysql.connect(
            user='root',
            passwd='1111',
            host='localhost',
            db='web_test'
        )
        cursor = sample_db.cursor()
        sql = "select no, 회사명, 회계년도, 총자산순이익률, 차입금의존도, 총자본영업이익률, 총자산영업이익률, 비유동장기적합률, 자기자본회전률, 자기자본순이익률," \
              " ocf_대_유동부채, 총자산회전율, 부도여부 from last10 limit 10;"
        cursor.execute(sql)
        result=cursor.fetchall()

        for list in result:
            date.append(list[2])
            a.append(list[3])
            aa = np.round(list[3] * 100, 3)
            bb = np.round(list[6] * 100, 3)
            cc = np.round(list[9], 3)
            dd = np.round(list[7], 3)

    finally:
        sample_db.close()

    return render_template('dash.html', result=result, _sum1=aa, _sum2=bb, _sum3=cc, _sum4=dd,
                           date=date, a=a)

@app.route('/model')
def model():
    return render_template('model.html')

@app.route('/modelsearch')
def modelsearch():
    _mname = request.args["_mname"]
    text1 = np.array(request.args["text1"], dtype=float) #총자산순이익률
    text2 = np.array(request.args["text2"], dtype=float) #총자산영업이익률
    text3 = np.array(request.args["text3"], dtype=float) #총자본영업이익률
    text4 = np.array(request.args["text4"], dtype=float) #자기자본순이익률
    text5 = np.array(request.args["text5"], dtype=float) #차입금의존도
    text6 = np.array(request.args["text6"], dtype=float) #비유동장기적합률
    text7 = np.array(request.args["text7"], dtype=float) #ocf 대 유동부채
    text8 = np.array(request.args["text8"], dtype=float) #자기자본회전율
    text9 = np.array(request.args["text9"], dtype=float) #총자산회전율

    modelsearch = np.array([text1, text5, text3, text2, text6, text8, text4, text7, text9])

    try:
        sample_db = pymysql.connect(
            user='root',
            passwd='1111',
            host='localhost',
            db='web_test'
        )
        cursor = sample_db.cursor()

        sql3 = "select 총자산순이익률, 차입금의존도, 총자본영업이익률, 총자산영업이익률, 비유동장기적합률, 자기자본회전률, 자기자본순이익률," \
              " ocf_대_유동부채, 총자산회전율, 부도여부 from last10;"

        cursor.execute(sql3)
        result2 = cursor.fetchall()

        final_col = pd.DataFrame(result2, columns=['총자산순이익률', '차입금의존도', '총자본영업이익률', '총자산영업이익률', '비유동장기적합률', '자기자본회전률', '자기자본순이익률', 'ocf 대 유동부채', '총자산회전율', '부도여부'])
        data_f = final_col

        X_train, X_test, y_train, y_test = train_test_split(data_f.drop(['부도여부'], axis=1), data_f['부도여부'],
                                                            test_size=0.2,
                                                            shuffle=True,
                                                            random_state=777,
                                                            stratify=data_f['부도여부'])
        X_train_rs = X_train.copy()
        X_test_rs = X_test.copy()

        for idx, i in enumerate(X_train.columns):
            globals()[f"rs_{idx}"] = RobustScaler()
            X_train_rs[i] = globals()[f"rs_{idx}"].fit_transform(X_train_rs[i].values.reshape(-1, 1))
            X_test_rs[i] = globals()[f"rs_{idx}"].transform(X_test_rs[i].values.reshape(-1, 1))

        sm = SMOTE(random_state=42, sampling_strategy=0.1)
        train_input_res3, train_target_res3 = sm.fit_resample(X_train_rs, y_train.ravel())

        dt = DecisionTreeClassifier(max_depth=5, random_state=42)
        dt.fit(train_input_res3, train_target_res3)

        lr_model = LogisticRegression(random_state=42)
        lr_model.fit(train_input_res3, train_target_res3)

        def rank(prob):
            if (prob <= 0.008278):
                rank = 1
            elif (prob <= 0.014232):
                rank = 2
            elif (prob <= 0.020253):
                rank = 3
            elif (prob <= 0.027612):
                rank = 4
            elif (prob <= 0.037594):
                rank = 5
            elif (prob <= 0.056411):
                rank = 6
            elif (prob <= 0.127776):
                rank = 7
            elif (prob <= 0.979940):
                rank = 8
            else:
                rank = 9

            return rank

        new_data = modelsearch
        for idx, i in enumerate(new_data):
            new_data[idx] = globals()[f"rs_{idx}"].transform(new_data[idx].reshape(-1, 1))

        score = dt.predict_proba(new_data.reshape(1,-1))[:,1]
        score1 = np.round(score * 100, 3)

        rank = rank(score)

    finally:
        sample_db.close()

    return render_template('model.html', score1=score1, rank=rank)


@app.route('/search')
def search():
    _name = request.args["_name"]
    date = []
    a = []
    _sum1 = 0
    _sum2 = 0
    _sum3 = 0
    _sum4 = 0
    try:
        sample_db = pymysql.connect(
            user='root',
            passwd='1111',
            host='localhost',
            db='web_test'
        )
        cursor = sample_db.cursor()
        sql = "select no, 회사명, 회계년도, 총자산순이익률, 차입금의존도, 총자본영업이익률, 총자산영업이익률, 비유동장기적합률, 자기자본회전률, 자기자본순이익률," \
              " ocf_대_유동부채, 총자산회전율, 부도여부 from last10 where 회사명 = %s;"

        val1 = (_name)
        cursor.execute(sql, val1)
        result = cursor.fetchall()

        for list in result:
            date.append(list[2])
            a.append(list[3])
            aa = np.round(list[3]*100, 3)
            bb = np.round(list[6]*100, 3)
            cc = np.round(list[9], 3)
            dd = np.round(list[7], 3)

    finally:
        sample_db.close()

    return render_template('dash.html', result=result, _sum1=aa, _sum2=bb, _sum3=cc, _sum4=dd,
                           date=date, a=a)

app.run