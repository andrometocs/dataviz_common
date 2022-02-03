import seaborn as sns
import os
import streamlit as st
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def main():
    """Common ML dataset explorer"""
    st.title("Machine Learning Dataset Explorer")
    st.subheader("Using streamlit app to explore Machine Learning Datasets")

    def file_Selector(folder_path='./datasets'):
        filenames = os.listdir(folder_path)
        selected_filename = st.selectbox('Select A file', filenames)
        return os.path.join(folder_path, selected_filename)

    filename = file_Selector()
    st.info('You selected {}'.format(filename))


if __name__ == "__main__":
    main()
