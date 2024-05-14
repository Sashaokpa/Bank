import streamlit as st
import pandas as pd
from pathlib import Path
from streamlit_pandas_profiling import st_profile_report
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import plotly.express as px


@st.cache_data
def get_melt_categorical(df, categorical_features):
    """Функция для отображения категориальных признаков"""
    # melt (расплавление)
    cat_df = pd.DataFrame(
        df[categorical_features].melt(
            var_name='column',
            value_name='value'
        ).value_counts()
    ).rename(columns={0: 'count'}).sort_values(by=['column', 'count'])
    return cat_df


@st.cache_data
def get_data_info(df):
    info = pd.DataFrame()
    info.index = df.columns
    info['Тип данных'] = df.dtypes
    info['Уникальных'] = df.nunique()
    info['Количество значений'] = df.count()
    return info


@st.cache_data
def get_profile_report(df):
    from pandas_profiling import ProfileReport
    pr = ProfileReport(df)
    return pr


@st.cache_data
def create_histogram(df, column_name):
    fig = px.histogram(
        df,
        x=column_name,
        marginal="box",
        color='y',
        title=f"Распределение {column_name}",
        template="plotly"
    )
    return fig


@st.cache_data
def create_correlation_matrix(df, numerical_features):
    corr = df[numerical_features].corr().round(2)
    fig = ff.create_annotated_heatmap(
        z=corr.values,
        x=list(corr.columns),
        y=list(corr.index),
        annotation_text=corr.values,
        colorscale='greens'

    )
    fig.update_layout(height=800)
    return fig


@st.cache_data
def get_simple_histograms(df, selected_category):
    fig = px.histogram(
        df,
        x=selected_category,
        color=selected_category,
        title=f'Распределение по {selected_category}'
    )
    return fig


@st.cache_data
def display_age_distribution(df):
    fig = px.box(
        df,
        x="education",
        y="age",
        color="education",
        title="Распределение возраста по уровням образования",
        labels={"age": "Возраст", "education": "Образование"}
    )
    fig.update_layout(height=700)
    return fig


@st.cache_data
def display_age_distribution_by_marital_status(df):
    fig = px.box(
        df,
        x='marital',
        y='age',
        color='y',
        title='Распределение возраста по семейному положению и откликам',
        facet_col='y',
        labels={'marital': 'Семейное положение', 'y': 'Отклик'}
    )
    fig.update_layout(height=700)
    return fig


@st.cache_data
def plot_categorical_features(df):
    plt.figure(figsize=(15, 15))

    for i, feature in enumerate(['job', 'marital', 'education', 'contact', 'month', 'y'], start=1):
        plt.subplot(3, 2, i)
        sns.countplot(data=df, x=feature, order=df[feature].value_counts().index)
        plt.title(f'Столбчатая диаграмма для {feature}')
        plt.xticks(rotation=45)

    plt.tight_layout()
    st.pyplot(plt, use_container_width=True)


@st.cache_data
def create_pairplot(df, selected_features, hue=None):
    sns.set_theme(style="whitegrid")
    pairplot_fig = sns.pairplot(
        df,
        vars=selected_features,
        hue=hue,
        palette='viridis',
        plot_kws={'alpha': 0.5, 's': 80, 'edgecolor': 'k'},
        height=3
    )
    plt.subplots_adjust(top=0.95)
    return pairplot_fig


@st.cache_data
def plot_average_duration(df):
    average_duration = df.groupby(['job', 'y'])['duration'].mean().reset_index()

    fig = px.bar(
        average_duration,
        x='job',
        y='duration',
        color='y',
        barmode='group',
        title='Средняя продолжительность контакта по типу работы и отклику',
        labels={'job': 'Тип работы', 'y': 'Отклик'}
    )
    fig.update_layout(height=500)
    return fig


@st.cache_data
def display_metrics(df):
    st.markdown("""
        В данном разделе представлены ключевые метрики, которые помогают оценить эффективность маркетинговых кампаний банка,
        а также понять поведение потенциальных клиентов при предложении открыть срочный вклад.
    """)
    # Расчет основных метрик
    average_age = int(df['age'].mean())
    success_rate = df['y'].value_counts(normalize=True)['yes'] * 100
    max_call_duration = df['duration'].max()
    average_contacts = df['campaign'].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Средний возраст", f"{average_age} лет")
    col2.metric("Процент успеха", f"{success_rate:.2f}%")
    col3.metric("Максимальная длительность звонка", f"{max_call_duration} секунд")
    col4.metric("Среднее количество контактов", f"{average_contacts:.1f}")


def display_box_plot(df, numerical_features, categorical_features):
    c1, c2, c3 = st.columns(3)
    feature1 = c1.selectbox('Первый признак', numerical_features, key='box_feature1')
    feature2 = c2.selectbox('Второй признак', categorical_features, key='box_feature2')
    filter_by = c3.selectbox('Фильтровать по', [None, *categorical_features], key='box_filter_by')

    if feature2 == filter_by:
        filter_by = None

    fig = px.box(
        df,
        x=feature1, y=feature2,
        color=filter_by,
        title=f"Распределение {feature1} по разным {feature2}",
    )
    fig.update_layout(height=900)
    st.plotly_chart(fig, use_container_width=True)


def app(df, current_dir: Path):
    st.title("Анализ прямых звонков из банка")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            ## Область применения
            Эта система анализа данных предназначена для использования в банковском секторе для повышения эффективности маркетинговых кампаний по прямым телефонным звонкам. Она позволяет предсказать, какие потенциальные клиенты скорее всего откликнутся на предложение открыть срочный вклад, что способствует более точному таргетингу и оптимизации ресурсов.
            В данном приложении рассматривается анализ данных в контексте банковского сектора, с фокусом на прогнозирование положительного отклика потенциальных клиентов банка на предложение открыть срочный вклад посредством прямых телефонных звонков.
            Такой подход позволяет банкам эффективно управлять маркетинговыми кампаниями, оптимизировать расходы и увеличивать клиентскую базу. Задача анализа данных заключается в идентификации ключевых факторов, влияющих на решение потенциального клиента оформить срочный вклад.
            """
        )
    with col2:
        st.image(str(current_dir / "images" / "main.webp"), use_column_width='auto')

    st.markdown("""
        ## Ключевые параметры и характеристики данных
        Для анализа использовался датасет, содержащий 41188 записей о контактах с клиентами, каждая из которых характеризуется 21 признаком, включая целевой. Перечень и описание признаков представлены ниже:
    """)
    tab1, tab2 = st.tabs(["Показать описание данных", "Показать пример данных"])
    with tab1:
        st.markdown(
            r"""
            ## Описание данных
            | Параметр                  | Описание                                                                                  |
            |---------------------------|-------------------------------------------------------------------------------------------|
            | age                       | Возраст потенциального клиента в годах                                                    |
            | job                       | Профессия (admin., blue-collar, entrepreneur, housemaid, management, retired, self-employed, services, student, technician, unemployed, unknown) |
            | marital                   | Семейное положение (divorced, married, single, unknown)                                   |
            | education                 | Образование (basic.4y, basic.6y, basic.9y, high.school, illiterate, professional.course, university.degree, unknown) |
            | default                   | Есть ли дефолт по кредиту (no, yes, unknown)                                              |
            | housing                   | Есть ли кредит на жильё (no, yes, unknown)                                                |
            | loan                      | Есть ли потребительский кредит (no, yes, unknown)                                         |
            | contact                   | Способ связи c потенциальным клиентом (cellular, telephone)                               |
            | month                     | Номер месяца, когда был крайний контакт с клиентом (jan, feb, mar, ..., nov, dec)         |
            | day_of_week               | День недели, когда был крайний контакт с клиентом (mon, tue, wed, thu, fri)               |
            | duration                  | Продолжительность крайнего звонка клиенту в секундах                                      |
            | campaign                  | Количество контактов с данным клиентом в течение текущей компании                         |
            | pdays                     | Количество дней, прошедшее с предыдущего контакта (999 для новых клиентов)                |
            | previous                  | Количество контактов с данным клиентом до текущей компании                                |
            | poutcome                  | Результат предыдущей маркетинговой компании (failure, nonexistent, success)               |
            | emp.var.rate              | Коэффициент изменения занятости - квартальный показатель                                  |
            | cons.price.idx            | Индекс потребительских цен - месячный показатель                                          |
            | cons.conf.idx             | Индекс доверия потребителей - ежемесячный показатель                                      |
            | euribor3m                 | Euribor 3-месячный курс - дневной индикатор                                               |
            | nr.employed               | Количество сотрудников - квартальный показатель                                           |
            | y                         | Подписал ли клиент срочный вклад (yes, no)                                                |
            """
        )
    with tab2:
        st.header("Пример данных")
        st.dataframe(df.head(50), height=600)

    categorical_features = df.select_dtypes(include='category').columns.tolist()
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    st.header("Предварительный анализ данных")
    st.dataframe(get_data_info(df), use_container_width=True)

    st.markdown("""
        Предварительный анализ данных показал следующее:
        * Как видим, данные полные, пропусков нет, поэтому нет необходимости заполнять пропуски.
        * Далее проверим данные на выбросы. Для начала сравним минимальное и максимальное значение со средним (для численных признаков):
    """)

    st.header("Основные статистики для признаков")
    display_metrics(df)

    tab1, tab2 = st.tabs(["Числовые признаки", "Категориальные признаки"])
    with tab1:
        st.header("Рассчитаем основные статистики для числовых признаков")
        st.dataframe(df.describe(), use_container_width=True)
        st.markdown(
            """
            Основные статистики для числовых признаков:
            * Средний возраст клиентов составляет примерно 40 лет, при этом минимальный и максимальный возраст - 17 и 98 лет соответственно.
            * Средняя продолжительность последнего контакта с клиентом (в секундах) - около 258, что свидетельствует о разнообразии взаимодействий.
            * Большинство клиентов (более 75%) не были контактированы до текущей кампании более одного раза (campaign).
            * Значение pdays (количество дней, прошедших после последнего контакта с клиентом из предыдущей кампании) для большинства записей составляет 999, что означает, что клиент не был ранее контактирован.
            * Индикаторы экономического контекста (emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed) показывают различную вариативность, что отражает изменения в экономических условиях.
            """
        )
    with tab2:
        st.header("Рассчитаем основные статистики для категориальных признаков")
        st.dataframe(df.describe(include='category'), use_container_width=True)
        st.markdown(
            """
            Количество уникальных значений для категориальных признаков:
            * Наибольшее количество уникальных значений наблюдается в столбцах job (12 различных профессий) и education (8 уровней образования).
            * Целевая переменная y (согласие клиента на депозитный продукт) имеет два уникальных значения: "да" и "нет".
        """)

    st.header("Визуализация данных")

    st.subheader("Визуализация числовых признаков")
    selected_feature = st.selectbox(
        "Выберите признак",
        numerical_features,
        key="create_histogram_selectbox1"
    )
    hist_fig = create_histogram(
        df,
        selected_feature
    )
    st.plotly_chart(hist_fig, use_container_width=True)

    st.markdown("""
        ## Гистограммы и ящики с усами
    
        Гистограмма — это вид диаграммы, представляющий распределение числовых данных. Она помогает оценить плотность вероятности распределения данных. Гистограммы идеально подходят для иллюстрации распределений признаков, таких как возраст клиентов или продолжительность контакта в секундах.
        
        Ящик с усами — это еще один тип графика для визуализации распределения числовых данных. Он показывает медиану, первый и третий квартили, а также "усы", которые простираются до крайних точек данных, не считая выбросов. Ящики с усами особенно полезны для сравнения распределений между несколькими группами и выявления выбросов.
    """)
    display_box_plot(
        df,
        numerical_features,
        categorical_features
    )

    tab1, tab2 = st.tabs(["Простые графики", "Показать отчет о данных"])
    with tab1:
        st.subheader("Распределение сотрудников")
        st.subheader("Столбчатые диаграммы для категориальных признаков")
        selected_category_simple_histograms = st.selectbox(
            'Категория для анализа',
            categorical_features,
            key='category_get_simple_histograms'
        )
        st.plotly_chart(get_simple_histograms(df, selected_category_simple_histograms), use_container_width=True)
        st.subheader("Распределение числовых признаков c группировкой по отклику")
        selected_feature = st.selectbox(
            "Выберите признак",
            numerical_features,
            key="create_histogram_selectbox2"
        )
        hist_fig = create_histogram(
            df,
            selected_feature
        )
        st.plotly_chart(hist_fig, use_container_width=True)

    with tab2:
        if st.button("Сформировать отчёт", use_container_width=True, type='primary'):
            st_profile_report(get_profile_report(df))

    @st.cache_data
    def plot_metrics(df, columns):
        plt.figure(figsize=(16, 20))
        for i, col in enumerate(columns, start=1):
            plt.subplot(len(columns), 2, 2 * i - 1)
            sns.histplot(data=df, x=col, hue="y", multiple="stack", kde=True)
            plt.title(f'Гистограмма для {col}')

            plt.subplot(len(columns), 2, 2 * i)
            sns.boxplot(data=df, x="y", y=col)
            plt.title(f'Ящик с усами для {col}')

        plt.tight_layout()
        st.pyplot(plt.gcf())  # Отображение созданного графика в Streamlit
        plt.clf()

    st.markdown("### Визуализация основных клиентских данных")
    plot_metrics(df, numerical_features[:5])
    st.markdown("""
        * Возраст (age): Распределение возраста показывает, что молодые люди и люди среднего возраста чаще откликаются на предложения банка. В ящиках с усами видно, что распределение возраста среди откликнувшихся и не откликнувшихся имеет некоторые различия, с более широким размахом у откликнувшихся.
        
        * Продолжительность последнего контакта (duration): Явный сдвиг к более длительным звонкам среди тех, кто согласился на вклад. Это подтверждается ящиком с усами, показывающим высокий размах и выбросы в данных по продолжительности звонков среди откликнувшихся.
        
        * Количество контактов в ходе текущей кампании (campaign): Большинство успешных контактов имеют меньшее количество попыток связи, что отображено в гистограмме и подтверждается ящиком с усами. Выбросы среди не откликнувшихся указывают на случаи с чрезмерным количеством попыток контакта.
        
        * Количество дней, прошедших с последнего контакта (pdays): Большая часть клиентов, не имеющих предыдущих контактов (значение 999 в датасете), склонна к отрицательному отклику, в то время как более короткие интервалы между контактами ассоциируются с положительным откликом.
        
        * Количество контактов до текущей кампании (previous): Клиенты с более высоким числом предыдущих контактов чаще давали положительный отклик, что видно по гистограммам и ящикам с усами.
    """)
    st.markdown("### Визуализация экономических показателей")
    plot_metrics(df, numerical_features[5:])
    st.markdown("""
        Показатели экономического контекста (emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed): Экономические показатели имеют значительное влияние на решение клиентов. Например, низкие ставки по еврибору (euribor3m) и низкий индекс потребительских цен (cons.price.idx) часто ассоциируются с большим количеством положительных откликов, что может отражать поиск более выгодных инвестиционных возможностей в условиях экономической нестабильности
    """)

    st.markdown(
        """
        ## Ящики с усами для числовых признаков
        Эти графики позволяют наглядно оценить распределение основных числовых параметров сотрудников в зависимости от их статуса увольнения.
       """
    )
    st.markdown("### Ящик с усами для возраста, разделенного по образованию")
    st.plotly_chart(display_age_distribution(df), use_container_width=True)
    st.markdown("""
        На графике представлено распределение возраста людей в зависимости от уровня их образования. Каждая коробка («ящик с усами») соответствует определённому уровню образования и показывает медиану (линия внутри коробки), 25-й и 75-й перцентили (границы коробки), а также выбросы (точки за пределами «усов»). 
        
        Категории образования:
        - базовое 4 года (basic.4y)
        - среднее (high_school)
        - базовое 6 лет (basic.6y)
        - базовое 9 лет (basic.9y)
        - профессиональное (professional.course)
        - неизвестное (unknown)
        - высшее (university.degree)
        - неграмотные (illiterate)
        
        Из графика видно, что медианный возраст увеличивается с уровнем образования от базового до высшего. Категория "неизвестное" образование показывает большой разброс возрастов. Уровень "неграмотные" имеет меньший медианный возраст, чем высшее образование, но выше, чем у некоторых базовых категорий. На каждом уровне образования присутствуют выбросы, что говорит о наличии людей с данным уровнем образования, которые значительно моложе или старше типичного возрастного диапазона.
    """)

    st.markdown("""
        ### Возраст, разделенный по семейному положению и откликам
    """)
    st.plotly_chart(display_age_distribution_by_marital_status(df), use_container_width=True)
    st.markdown("""
        Ящики с усами показывают, что распределение возраста клиентов банка варьируется в зависимости от их семейного положения и реакции на предложение о срочном вкладе. Женатые клиенты в среднем старше и показывают большую склонность к положительному отклику по сравнению с неженатыми/незамужними, что может свидетельствовать о более консервативных финансовых предпочтениях или желании обеспечить финансовую стабильность. 
        Молодые и одинокие клиенты также демонстрируют интерес к срочным вкладам, но их отклики несколько менее выражены. Разведенные клиенты часто находятся в более зрелом возрасте и могут видеть в срочных вкладах возможность управления своими средствами после изменения семейного статуса. В категории с неизвестным семейным положением обнаруживается широкий возрастной диапазон и высокий уровень разброса, что усложняет формирование определенных выводов из-за потенциально небольшого размера выборки или недостатка данных.
    """)

    st.markdown(
        """
        ## Столбчатые диаграммы для категориальных признаков
        """
    )
    plot_categorical_features(df)
    st.markdown("""
        **Работа (Job):** Самые высокие количества занятости приходятся на категории 'admin.' и 'blue-collar'. Это может указывать на то, что эти группы являются основными целями маркетинговых кампаний банка.
        
        **Семейное положение (Marital):** Большинство клиентов состоят в браке, за ними следуют одиночки и разведённые. Это распределение может быть связано с финансовой стабильностью или потребностями, которые меняются с изменением семейного положения.
        
        **Образование (Education):** Больше всего клиентов с высшим образованием, что может отражать высокую долю людей с высшим образованием среди клиентской базы банка. Категории 'high.school' и 'basic.9y' также хорошо представлены.
        
        **Способ связи (Contact):** Существенно больше клиентов были контактированы через мобильный телефон по сравнению с городским телефоном, что подчеркивает популярность и удобство мобильных устройств.
        
        **Месяц (Month):** Наибольшее количество контактов приходится на месяцы май, июль и август, что может указывать на сезонные пики активности банковских кампаний.
    """)

    st.markdown("## Средняя продолжительность контакта по типу работы и отклику")
    st.plotly_chart(plot_average_duration(df), use_container_width=True)
    st.markdown("""
        График показывает сравнение средней продолжительности контактов по различным типам работы, разделённых на две категории: контакты, приведшие к отклику (да), и контакты, не приведшие к отклику (нет). Столбцы представлены для разных профессий, таких как административный персонал (admin.), рабочие (blue-collar), предприниматели (entrepreneur) и так далее.
    
        Из графика видно, что в каждой профессиональной группе средняя продолжительность контакта, которая привела к отклику (да), была выше, чем контактов, которые не привели к отклику (нет). Наибольшая средняя продолжительность контакта, приводящего к отклику, наблюдается у предпринимателей (entrepreneur), в то время как самая низкая — у студентов (student). Это может указывать на то, что более длительные разговоры могут быть более эффективными для достижения положительного результата или отклика.
    """)

    st.header("Корреляционный анализ")
    st.plotly_chart(create_correlation_matrix(df, numerical_features), use_container_width=True)
    st.markdown("""
        Матрица корреляции позволяет определить связи между признаками. Значения в матрице колеблются от -1 до 1, где:
        
        - 1 означает положительную линейную корреляцию,
        - -1 означает отрицательную линейную корреляцию,
        - 0 означает отсутствие линейной корреляции.
        
        Корреляционная матрица представляет связь между различными числовыми параметрами. В данном случае:
        
        Интерпретация корреляционной матрицы:
        
        * emp.var.rate (уровень изменения занятости) и euribor3m имеют очень высокую корреляцию (0.97), что говорит о тесной связи между уровнем занятости и европейской процентной ставкой.
        * Аналогично, euribor3m и nr.employed (количество занятых) также сильно коррелируют (0.95), подтверждая связь между процентными ставками и уровнем занятости.
        * emp.var.rate и cons.price.idx (индекс потребительских цен) имеют значительную корреляцию (0.78), указывая на влияние экономических условий на потребительские цены.
        * pdays (количество дней, прошедших после последнего контакта) и previous (количество контактов до текущей кампании) имеют отрицательную корреляцию (-0.59), что может отражать стратегию обращения к клиентам, с которыми не контактировали некоторое время.
        
        
        Как видно из тепловой карты euribor3m и nr.employed сильно коррелируют с emp.var.rate, впоследствии, на этапе отбора признаков мы их удалим, когда будем отбирать признаки.
    """)

    st.markdown(
        """
        ## Точечные диаграммы для пар числовых признаков
        """
    )
    selected_features = st.multiselect(
        'Выберите признаки',
        numerical_features,
        default=['age', 'duration', 'campaign', 'previous', 'euribor3m'],
        key='pairplot_vars'
    )

    # Опциональный выбор категориальной переменной для цветовой дифференциации
    hue_option = st.selectbox(
        'Выберите признак для цветового кодирования (hue)',
        ['None'] + categorical_features,
        index=11,
        key='pairplot_hue'
    )
    if hue_option == 'None':
        hue_option = None
    if selected_features:
        st.pyplot(create_pairplot(df, selected_features, hue=hue_option))
    else:
        st.error("Пожалуйста, выберите хотя бы один признак для создания pairplot.")
