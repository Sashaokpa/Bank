import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from pathlib import Path
from apps import home, prediction_logreg
from plotly.io import templates

templates.default = "ggplot2"

st.set_page_config(
    page_title="Анализ прямых звонков из банка",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def preprocess_data(file_path):
    df = pd.read_csv(file_path, sep=';')

    # Преобразование столбцов к категориальным типам
    categorical_columns = [
        'job',
        'marital',
        'education',
        'default',
        'housing',
        'loan',
        'contact',
        'month',
        'day_of_week',
        'poutcome',
        'y'
    ]
    for column in categorical_columns:
        df[column] = df[column].astype('category')
    return df


class Menu:
    apps = [
        {
            "func": home.app,
            "title": "Главная",
            "icon": "house-fill"
        },
        {
            "func": prediction_logreg.app,
            "title": "Прогнозирование",
            "icon": "person-check-fill"
        },
    ]

    def run(self):
        with st.sidebar:
            titles = [app["title"] for app in self.apps]
            icons = [app["icon"] for app in self.apps]

            selected = option_menu(
                "Menu",
                options=titles,
                icons=icons,
                menu_icon="cast",
                default_index=0,
            )
            st.info(
                """
                ## Анализ прямых звонков из банка
                Эта система анализа данных предназначена для банковского сектора с целью повышения эффективности маркетинговых кампаний по прямым телефонным звонкам. Она предсказывает, какие клиенты скорее всего откликнутся на предложение открыть срочный вклад, что способствует более точному таргетингу и оптимизации ресурсов.
                """
            )
        return selected


if __name__ == '__main__':
    current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()

    df = preprocess_data(current_dir / 'bank-additional-full.csv')
    categorical_features = df.select_dtypes(include='category').columns.tolist()
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    menu = Menu()
    st.sidebar.image(str(current_dir / 'images' / 'logo.png'))
    selected = menu.run()
    for app in menu.apps:
        if app["title"] == selected:
            app["func"](df, current_dir)
            break
