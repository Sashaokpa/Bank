import joblib
from pathlib import Path
import pandas as pd
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from streamlit_option_menu import option_menu


@st.cache_data
def create_plot_roc_curve(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name='ROC кривая (AUC = %0.2f)' % roc_auc_score(y_true, y_prob)
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        line=dict(dash='dash'),
        name='Диагональ'
    ))

    fig.update_layout(
        xaxis_title='Доля ложно-положительных результатов',
        yaxis_title='Доля истинно-положительных результатов',
        title='Кривая ROC',
        width=900
    )
    return fig


@st.cache_data
def create_plot_confusion_matrix(y_true, y_pred, normalize=False):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = cm.round(3)

    fig = px.imshow(
        cm,
        labels=dict(x="Предсказанный класс", y="Истинный класс"),
        x=['Нет', 'Да'], y=['Нет', 'Да'],
        title='Нормализованная матрица ошибок' if normalize else 'Матрица ошибок',
        color_continuous_scale='Blues'
    )

    fig.update_xaxes(tickangle=45, tickmode='array', tickvals=np.arange(2), ticktext=['Нет', 'Да'])
    fig.update_yaxes(tickangle=45, tickmode='array', tickvals=np.arange(2), ticktext=['Нет', 'Да'])
    # Добавление надписей
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            fig.add_annotation(
                x=i, y=j,
                text=str(cm[j, i]),
                showarrow=False,
                font=dict(color="white" if cm[j, i] > thresh else "black"),
                align="center"
            )
    return fig


def score(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        pred = clf.predict(X_train)
        prob = clf.predict_proba(X_train)[:, 1]
        clf_report = classification_report(y_train, pred, output_dict=True)
        st.subheader("Результат обучения:")
        st.write(f"Точность модели: {accuracy_score(y_train, pred) * 100:.2f}%")
        st.write("ОТЧЕТ ПО КЛАССИФИКАЦИИ:", pd.DataFrame(clf_report).transpose())
        st.plotly_chart(create_plot_roc_curve(y_train, prob), use_container_width=True)
        st.plotly_chart(create_plot_confusion_matrix(y_train, pred, normalize=True), use_container_width=True)
    else:
        pred = clf.predict(X_test)
        prob = clf.predict_proba(X_test)[:, 1]
        clf_report = classification_report(y_test, pred, output_dict=True)
        st.subheader("Результат тестирования:")
        st.write(f"Точность модели: {accuracy_score(y_test, pred) * 100:.2f}%")
        st.write("ОТЧЕТ ПО КЛАССИФИКАЦИИ:", pd.DataFrame(clf_report).transpose())
        st.plotly_chart(create_plot_roc_curve(y_test, prob), use_container_width=True)
        st.plotly_chart(create_plot_confusion_matrix(y_test, pred, normalize=True), use_container_width=True)


def print_model_adequacy_section(current_dir: Path):
    st.markdown(
        """
        ## Оценка адекватности модели
        При оценки адекватности модели важно использовать несколько метрик, которые помогают оценить различные аспекты производительности модели.
 
        ### Матрица ошибок (Confusion Matrix)
 
        Матрица ошибок позволяет визуально оценить, как модель справляется с каждым из классов задачи. Она показывает, сколько примеров, предсказанных в каждом классе, действительно принадлежат этому классу.
        """
    )
    st.image(str(current_dir / 'images' / 'matrix.jpg'))
    st.markdown(
        """
        ### Отчет о классификации (Precision, Recall, F1-Score)
        * Precision (Точность) описывает, какая доля положительных идентификаций была верной (TP / (TP + FP)).
        * Recall (Полнота) показывает, какая доля фактических положительных классов была идентифицирована (TP / (TP + FN)).
        * F1-Score является гармоническим средним Precision и Recall и помогает учесть обе эти метрики в одной.

        ### Кривая ROC и площадь под кривой AUC
        * ROC кривая (Receiver Operating Characteristic curve) помогает визуально оценить качество классификатора. Ось X показывает долю ложноположительных результатов (False Positive Rate), а ось Y — долю истинноположительных результатов (True Positive Rate).
        * AUC (Area Under Curve) — площадь под ROC кривой, которая дает количественную оценку производительности модели.
        """
    )


@st.cache_data
def Information_Value(x, y):
    # Функция расчёта Information Value
    df = pd.DataFrame({'x': x, 'y': y})
    good = df.groupby('x')['y'].sum() / np.sum(df['y'])
    bad = (df.groupby('x')['y'].count() - df.groupby('x')['y'].sum()) / (len(df['y']) - np.sum(df['y']))
    WOE = np.log((good + 0.000001) / bad)
    IV = (good - bad) * WOE
    return IV.sum()


@st.cache_data
def compute_iv_for_features(df, target_column):
    iv_values = {feature: Information_Value(df[feature], df[target_column]) for feature in df.columns if
                 feature != target_column}
    return iv_values


def encode_features(df, features):
    encoders = {}
    for feature in features:
        le = LabelEncoder()
        df[feature] = le.fit_transform(df[feature])
        encoders[feature] = le
    return df, encoders


def app(df, current_dir: Path):
    st.title("Анализ прямых звонков из банка")

    st.image(str(current_dir / "images" / "main2.webp"), width=150, use_column_width='auto')

    df_encoded = df.copy(deep=True)
    st.markdown(
        """
        # Подготовка набора данных
        Перед подачей наших данных в модель машинного обучения нам сначала нужно подготовить данные. Это включает в себя кодирование всех категориальных признаков (либо LabelEncoding, либо OneHotEncoding), поскольку модель ожидает, что признаки будут представлены в числовой форме. Также для лучшей производительности мы выполним масштабирование признаков, то есть приведение всех признаков к одному масштабу с помощью StandardScaler, предоставленного в библиотеке scikit-learn.
        """
    )
    st.title("Преобразование категориальных переменных")

    st.markdown("""
        ### Преобразование категориальных переменных

        Сделаем кодирование категориальных переменных с помощью LabelEncoder из библиотеки Scikit-Learn. Это преобразует категории в числа, что необходимо для использования этих данных в большинстве алгоритмов машинного обучения.

        Целевая переменная `y` кодируется как 1 для "yes" и 0 для "no". Это стандартный подход для бинарной классификации в машинном обучении, где нужно предсказать одно из двух состояний.

        #### Label Encoding

        Формула кодирования конкретной категории $C$ в число может быть представлена как: $ \\text{code}(C) = i $
        где $i$ — порядковый номер категории $C$ в данных.

        #### Общие формулы и принципы

        В процессе предобработки данных могут использоваться следующие операции и принципы:

        - **Нормализация:** приведение всех числовых переменных к единому масштабу, чтобы улучшить сходимость алгоритма. Обычно используется Min-Max scaling или Z-score стандартизация.
        - **One-hot Encoding:** преобразование категориальных переменных в бинарные векторы; применяется, когда порядок категорий не имеет значения.
        - **Отбор признаков:** удаление нерелевантных или малоинформативных признаков для упрощения модели и улучшения её обобщающей способности.

        Эти методы помогают подготовить данные для эффективного обучения моделей машинного обучения.
        """)

    categorical_features = df.select_dtypes(include='category').columns.tolist()
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    df_encoded = df.copy(deep=True)
    df_encoded.drop(['euribor3m', 'nr.employed'], axis=1, inplace=True)
    df_encoded.drop(['duration'], axis=1, inplace=True)
    categorical_features.remove('y')
    df_encoded, encoders = encode_features(df_encoded, categorical_features)
    df_encoded['y'] = df_encoded['y'].map({'yes': 1, 'no': 0})
    df_encoded['y'] = df_encoded['y'].astype(int)
    df_encoded = df_encoded.reindex(sorted(df_encoded.columns), axis=1)

    st.markdown(
        """
        ## Кодирование признаков (Feature Encoding)
        """
    )
    feature_encoding_tab1, feature_encoding_tab2 = st.tabs([
        "Данные до Feature Encoding",
        "Данные после Feature Encoding"
    ])
    with feature_encoding_tab1:
        st.dataframe(df.head())
    with feature_encoding_tab2:
        st.dataframe(df_encoded.head())

    st.markdown("""
     ### Information Value (IV)

     **Information Value (IV)** - это метрика, используемая для оценки предиктивной способности или силы признака в контексте бинарной классификации. Эта метрика помогает определить, насколько хорошо переменная может различать две группы (например, дефолт/не дефолт в кредитном скоринге). IV рассчитывается на основе Weight of Evidence (WOE), которая была рассчитана на предыдущих шагах.

     #### Формула для расчёта IV:

     $$ IV = \\sum_{i} (\\text{Good}_i - \\text{Bad}_i) \\times \\text{WOE}_i $$

     где:
     - $Good_i$ - доля "хороших" клиентов (например, тех, кто своевременно выплачивает кредит) в группе $i$,
     - $Bad_i$ - доля "плохих" клиентов в той же группе,
     - $WOE_i$ - Weight of Evidence для группы $i$.

     ### Применение и интерпретация IV

     - **Низкое значение IV** (меньше 0.1) указывает, что переменная имеет очень слабую предиктивную способность.
     - **Среднее значение IV** (0.1 до 0.3) говорит о средней предиктивной способности.
     - **Высокое значение IV** (больше 0.3) показывает, что переменная является сильным предиктором.

     ### Выводы по IV признаков

     - Признаки с очень высоким IV, такие как `emp.var.rate`, `cons.price.idx`, и `cons.conf.idx` (IV > 1), являются очень мощными предикторами и должны быть включены в модель.
     - Признаки с IV близким к 0, как `loan` и `day_of_week`, имеют низкую предиктивную способность по отношению к целевой переменной.

     Эти инсайты помогают в оптимизации и улучшении моделей машинного обучения, особенно в задачах кредитного скоринга и риск-менеджмента, позволяя фокусироваться на наиболее информативных переменных.

     Посмотрим на Information Value категориальных признаков:
     """)

    iv_values = compute_iv_for_features(df_encoded, 'y')
    for feature, iv in iv_values.items():
        st.write(f'IV для {feature} = {iv:.4f}')
    st.markdown(
        'Сравнивая Information value, видим, что признаки housing, loan и day_of_week являются неинформативными, поэтому мы их можем удалить. Кстати говоря, это же мы видели, когда строили графики, показывающие долю положительных откликов в зависимости от значения признака.')

    st.subheader('Разделение данных')
    X = df_encoded.drop('y', axis=1).copy(deep=True)
    Y = df_encoded['y']

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    st.write("Размер тренировочных данных:", X_train.shape, y_train.shape)
    st.write("Размер тестовых данных:", X_test.shape, y_test.shape)

    tab1, tab2 = st.tabs(["Тренировочные данные", "Тестовые данные"])

    with tab1:
        st.subheader("Тренировочные данные")
        st.markdown("""
          **Описание:** Тренировочные данные используются для подгонки модели и оценки её параметров.
          Эти данные получены путем исключения из исходного датасета столбцов с целевой переменной 'y'.

          **Данные тренировочного набора (X_train)**.
          Обучающий набор данных содержит информацию о признаках, используемых для обучения модели.
          """)
        st.dataframe(X_train.head(15), use_container_width=True)
        st.markdown("""
          **Целевая переменная (y_train)**.
          Целевая переменная содержит значения цены, которые модель должна научиться прогнозировать.
          В качестве целевой переменной для тренировочного набора используются исключительно значения столбца 'y'.
          """)
        st.dataframe(pd.DataFrame(y_train.head(15)).T)

    with tab2:
        st.subheader("Тестовые данные")
        st.markdown("""
          **Описание:** Тестовые данные используются для проверки точности модели на данных, которые не участвовали в тренировке.
          Это позволяет оценить, как модель будет работать с новыми, ранее не виденными данными.
          """)
        st.markdown("""
          **Данные тестового набора (X_test)**.
          Тестовый набор данных содержит информацию о признаках, используемых для оценки модели.
          """)
        st.dataframe(X_test.head(15), use_container_width=True)
        st.markdown("""
          **Целевая переменная (y_test)**.
          Целевая переменная представляет собой значения, которые модель пытается предсказать.
          """)
        st.dataframe(pd.DataFrame(y_test.head(15)).T)

    st.markdown(
        """
        # Моделирование
        ## Работа с несбалансированными данными
        Обратите внимание, что у нас есть несбалансированный набор данных, в котором большинство наблюдений относятся к одному типу ('NO'). В нашем случае, например, примерно 84% наблюдений имеют метку 'No', а только 16% - 'Yes', что делает этот набор данных несбалансированным.
        
        Для работы с такими данными необходимо принять определенные меры, иначе производительность нашей модели может существенно пострадать. В этом разделе я рассмотрю два подхода к решению этой проблемы.
        
        ### Увеличение числа примеров меньшинства или уменьшение числа примеров большинства
        В несбалансированных наборах данных основная проблема заключается в том, что данные сильно искажены, т.е. количество наблюдений одного класса значительно превышает количество наблюдений другого. Поэтому в этом подходе мы либо увеличиваем количество наблюдений для класса-меньшинства (oversampling), либо уменьшаем количество наблюдений для класса-большинства (undersampling).
        
        Стоит отметить, что в нашем случае количество наблюдений и так довольно мало, поэтому более подходящим будет метод увеличения числа примеров.
        
        Ниже я использовал технику увеличения числа примеров, известную как SMOTE (Synthetic Minority Oversampling Technique), которая случайным образом создает некоторые "синтетические" инстансы для класса-меньшинства, чтобы данные по обоим классам стали более сбалансированными.
        
        Важно использовать SMOTE до шага кросс-валидации, чтобы избежать переобучения модели, как это бывает при выборе признаков.
        
        ###  Выбор правильной метрики оценки
        Еще один важный аспект при работе с несбалансированными классами - это выбор правильных оценочных метрик.
        
        Следует помнить, что точность (accuracy) не является хорошим выбором. Это связано с тем, что из-за искажения данных даже алгоритм, всегда предсказывающий класс-большинство, может показать высокую точность. Например, если у нас есть 20 наблюдений одного типа и 980 другого, классификатор, предсказывающий класс-большинство, также достигнет точности 98%, но это не будет полезной информацией.
        
        В таких случаях мы можем использовать другие метрики, такие как:
        
        - **Точность (Precision)** — (истинно положительные)/(истинно положительные + ложно положительные)
        - **Полнота (Recall)** — (истинно положительные)/(истинно положительные + ложно отрицательные)
        - **F1-Score** — гармоническое среднее точности и полноты
        - **AUC ROC** — ROC-кривая, график между чувствительностью (Recall) и (1-specificity) (Специфичность=Точность)
        - **Матрица ошибок** — отображение полной матрицы ошибок
        """
    )

    st.markdown(
        r"""
        ### Логистическая регрессия
        Логистическая регрессия — это статистический метод анализа, используемый для моделирования зависимости дихотомической переменной (целевой переменной с двумя возможными исходами) от одного или нескольких предикторов (независимых переменных). Основное отличие логистической регрессии от линейной заключается в том, что первая предсказывает вероятность наступления события, используя логистическую функцию, что делает её идеальной для классификационных задач.
        
        #### Математическая модель
        
        ##### Логистическая функция (Сигмоид)
        Основой логистической регрессии является логистическая функция, также известная как сигмоид. Она описывается следующей формулой:
        
        $$
        P(y=1|X) = \frac{1}{1 + e^{-z}} 
        $$
        
        где $ z = \beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_nx_n $ — линейная комбинация входных переменных $ X $ (независимых переменных) и коэффициентов модели $ \beta $ (включая свободный член $ \beta_0 $ и коэффициенты при переменных $ \beta_1, \beta_2, ..., \beta_n $).
        
        ##### Интерпретация коэффициентов
        Коэффициенты в логистической регрессии интерпретируются через шансы (odds) и логарифм шансов:
        
        - **Шансы**: Вероятность того, что событие произойдет, деленная на вероятность того, что событие не произойдет.
        
        $$ \text{odds} = \frac{P(y=1|X)}{1 - P(y=1|X)} = e^z $$
        
        - **Логарифм шансов**:
        
        $$\log(\text{odds}) = z = \beta_0 + \beta_1x_1 + ... + \beta_nx_n $$
        
        Каждый коэффициент $\beta_i $ показывает, как изменится логарифм шансов, если соответствующая переменная увеличится на одну единицу, при условии, что все остальные переменные остаются неизменными.
        
        #### Регуляризация
        
        Регуляризация используется для предотвращения переобучения путем добавления штрафа за слишком большие значения коэффициентов к функции потерь:
        
        - **L1-регуляризация (Lasso)**:
        
          Штрафует сумму абсолютных значений коэффициентов. Это может привести к обнулению некоторых коэффициентов, что делает модель разреженной и может помочь в отборе признаков.
        
        - **L2-регуляризация (Ridge)**:
        
          Штрафует сумму квадратов коэффициентов, что предотвращает их слишком большое увеличение и помогает уменьшить переобучение, но не делает модель разреженной.
        
        #### Оценка модели
        
        Для оценки качества модели логистической регрессии часто используют ROC AUC (Area Under the Receiver Operating Characteristics Curve). Эта метрика
        
         помогает оценить, насколько хорошо модель может различать два класса (например, положительный и отрицательный).
        
        - **ROC AUC**:
          - Чем ближе значение ROC AUC к 1, тем лучше модель различает два класса.
          - Значение 0.5 говорит о том, что модель работает не лучше случайного гадания.
        
        #### Параметры в Логистической Регрессии:
        
        - **Penalty**: Тип регуляризации, используемый в модели (обычно `l1` или `l2`). Регуляризация помогает предотвратить переобучение модели путем штрафа за слишком большие коэффициенты.
          - `l1` (Lasso регуляризация): Штрафует абсолютное значение коэффициентов и может обнулять некоторые из них, делая модель разреженной.
          - `l2` (Ridge регуляризация): Штрафует квадраты коэффициентов, предотвращает их быстрый рост.
        
        - **C**: Инверсия силы регуляризации (`C = 1/λ`). Меньшие значения `C` указывают на сильнее регуляризацию.
        
        - **Class_weight**: Веса классов. Этот параметр используется, если классы в данных имеют различное количество образцов или если один класс более важен другого. `None` означает, что все классы имеют одинаковый вес, `balanced` автоматически взвешивает классы в соответствии с частотой их встречаемости.
        
        #### GridSearchCV
        
        `GridSearchCV` — это метод, который позволяет систематически просматривать множество комбинаций параметров, выбирая те, которые максимизируют качество модели на основе заданной метрики (в вашем случае — `roc_auc`).
        
        #### Процесс:
        1. **Задание параметров для поиска**: Создается словарь параметров, которые должны быть проверены.
        2. **Настройка кросс-валидации**: Определение метода кросс-валидации (здесь `StratifiedKFold`), который сохраняет соотношение классов в каждом фолде.
        3. **Подбор модели**: `LogisticRegression` подается в `GridSearchCV` вместе с параметрами и методом валидации.
        4. **Обучение**: `GridSearchCV` автоматически обучает модели на каждой комбинации параметров и валидирует их по стратегии кросс-валидации.
        
        #### Результаты:
        - **Best parameters**: Наилучшие параметры (`{'C': 0.02, 'class_weight': None, 'penalty': 'l2'}`), найденные в результате поиска.
        - **ROC_AUC score**: Наивысшее значение ROC-AUC (0.7958), которое показывает эффективность модели в различении двух классов (`yes` и `no`). Это значение близко к 1, что указывает на высокую предсказательную способность модели.
        """
    )
    try:
        def test1():
            return joblib.load(str(current_dir / "models" / 'logistic_regression_model.joblib'))

        log_reg = test1()
    except Exception as e:
        log_reg = LogisticRegression(C=0.01, penalty='l2', solver='liblinear')
        log_reg.fit(X_train, y_train)
        joblib.dump(log_reg, str(current_dir / "models" / 'logistic_regression_model.joblib'))

    print_model_adequacy_section(current_dir)
    tab1, tab2 = st.tabs(["Результаты модели на зависимых данных", "Результаты модели на независимых данных", ])
    with tab1:
        score(log_reg, X_train, y_train, X_test, y_test, train=True)
    with tab2:
        score(log_reg, X_train, y_train, X_test, y_test, train=False)

    st.markdown(
        """
        ### Результаты
        
        На основе представленных результатов обучения и тестирования модели логистической регрессии, а также визуализаций ROC кривой и нормализованной матрицы ошибок, можно сделать следующие выводы:
        
        1. **Кривая ROC**:
        - Значения AUC (Area Under Curve) для обеих кривых ROC близки к 0.8, что указывает на довольно высокую способность модели различать классы. Это хороший результат, поскольку AUC находится далеко от 0.5, что означало бы отсутствие дискриминационной способности у модели. Кривая ROC значительно выше диагонали, что также указывает на эффективность модели.
        
        2. **Матрица ошибок**:
        - Нормализованные матрицы ошибок для обучающего и тестового набора данных показывают, что модель имеет высокий процент истинно положительных результатов (True Positive Rate) по отношению к негативному классу, что соответствует высокому показателю Recall для класса 0.
        - Однако Recall для класса 1 (положительного класса) остается низким, что указывает на то, что значительное количество положительных случаев ошибочно классифицированы как негативные (False Negative).
        
        3. **Отчет по классификации**:
        - Точность модели (Accuracy) довольно высока как для обучающего (89.90%), так и для тестового (90.23%) набора данных.
        - Precision для положительного класса (1) ниже, чем для негативного класса (0), что может указывать на осторожность модели в предсказании положительного класса.
        - Recall для положительного класса значительно ниже, чем для негативного, что подтверждает наблюдения из матрицы ошибок.
        
        4. **Интерпретация результатов**:
        - Модель демонстрирует хорошее общее качество предсказаний, но имеет тенденцию к лучшему выявлению негативных исходов (класс 0) в ущерб положительным (класс 1).
        - В контексте, например, кредитного скоринга это означало бы, что модель более консервативна и склонна к минимизации риска, за счет увеличения количества ложно-отрицательных результатов, тем самым возможно упуская потенциально надежных клиентов.
        - Возможно, потребуется пересмотреть порог классификации или внести корректировки в процесс подготовки данных и отбора признаков для улучшения предсказательной способности модели по отношению к положительному классу.
        """
    )

    df['predicted_prob'] = log_reg.predict_proba(X)[:, 1]

    # Filter data points with probability of positive response > 80%
    high_prob_customers = df[df['predicted_prob'] > 0.85]

    #
    st.write(high_prob_customers)
    with st.form("Ввод данных клиента"):
        st.subheader('Введите параметры для прогноза')

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('### Основные характеристики')
            age = st.number_input("Возраст", min_value=17, max_value=120, value=35)
            job = st.selectbox("Профессия", options=list(encoders['job'].classes_))
            education = st.selectbox("Образование", options=list(encoders['education'].classes_))
            marital = option_menu(
                "Семейное положение",
                options=list(encoders['marital'].classes_),
                menu_icon="house-heart",
                icons=["person", "people", "person-x", "question-circle"],
                default_index=0
            )

            st.markdown('### Финансовое состояние')
            default = option_menu(
                "Долг",
                options=list(encoders['default'].classes_),
                menu_icon="warning",
                icons=["check-circle", "exclamation-circle", "question-circle"],
                default_index=0
            )
            housing = option_menu(
                "Ипотека",
                options=list(encoders['housing'].classes_),
                menu_icon="home",
                icons=["check-circle", "exclamation-circle", "question-circle"],
                default_index=0
            )
            loan = option_menu(
                "Кредит",
                options=list(encoders['loan'].classes_),
                menu_icon="credit-card",
                icons=["check-circle", "exclamation-circle", "question-circle"],
                default_index=0
            )

        with col2:
            st.markdown('### Контактная информация')
            contact = option_menu(
                "Контакт",
                options=sorted(encoders['contact'].classes_),
                menu_icon="phone",
                icons=["phone", "telephone-plus"],
                default_index=0
            )
            month = st.selectbox("Месяц", options=list(encoders['month'].classes_))
            day_of_week = st.selectbox("День недели", options=list(encoders['day_of_week'].classes_))
            duration = st.number_input("Продолжительность (сек)", min_value=0, max_value=5000, value=300)
            campaign = st.number_input("Кампании", min_value=1, max_value=100, value=1)
            pdays = st.number_input("Прошедшие дни", min_value=0, max_value=999, value=4)
            previous = st.number_input("Предыдущие", min_value=0, max_value=10, value=3)
            poutcome = option_menu(
                "Результат кампании",
                options=list(encoders['poutcome'].classes_),
                menu_icon="trophy",
                icons=["emoji-frown", "emoji-neutral", "emoji-smile"],
                default_index=0
            )

        st.markdown('### Экономические показатели')
        emp_var_rate = st.number_input("Уровень занятости", value=-2.9)
        cons_price_idx = st.number_input("Индекс потребительских цен", value=92.2)
        cons_conf_idx = st.number_input("Индекс доверия потребителей", value=-31.4)
        euribor3m = st.number_input("Euribor 3m", value=0.883)
        nr_employed = st.number_input("Количество сотрудников", value=5000)

        if st.form_submit_button("Прогнозировать", type='primary', use_container_width=True):
            inputs = {
                'age': age, 'job': job, 'marital': marital, 'education': education, 'default': default,
                'housing': housing, 'loan': loan, 'contact': contact, 'month': month,
                'day_of_week': day_of_week, 'duration': duration, 'campaign': campaign, 'pdays': pdays,
                'previous': previous, 'poutcome': poutcome, 'emp.var.rate': emp_var_rate,
                'cons.price.idx': cons_price_idx,
                'cons.conf.idx': cons_conf_idx, 'euribor3m': euribor3m, 'nr.employed': nr_employed
            }
            for col in categorical_features:
                inputs[col] = encoders[col].transform([inputs[col]])[0]

            input_df = pd.DataFrame([inputs])
            input_df = input_df.reindex(sorted(X_test.columns), axis=1)

            # Here you would make a prediction using your model, for demonstration we will use a dummy prediction
            prediction = log_reg.predict_proba(input_df)[:, 1][0]
            if prediction < 0.4:
                st.warning(f"Прогноз успешно выполнен! Вероятность положительного отклика: {prediction:.2f}")
                color = 'darkorange'
            else:
                st.success(f"Прогноз успешно выполнен! Вероятность положительного отклика: {prediction:.2f}")
                color = 'darkgreen'

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction,
                gauge={'axis': {'range': [0, 1]}, 'bar': {'color': color}},
                title={"text": "Прогнозируемая вероятность"}
            ))

            fig.update_layout(paper_bgcolor="#f0f2f6", font={'color': color, 'family': "Arial"})
            st.plotly_chart(fig, use_container_width=True)
