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
    st.title("EDA & Machine Learning Dataset Explorer")
    st.subheader("Using streamlit app to explore Machine Learning Datasets")

    st.sidebar.subheader("Upload Files and Visualization settings")

    def save_uploadedFile(uploadedFile):
        with open(os.path.join("./datasets", uploadedFile.name), "wb") as f:
            f.write(uploadedFile.getBuffer())
        return st.success("Saved file: {} to datasets".format(uploadedFile.name))

    datafile = st.sidebar.file_uploader(
        "Upload CSV", type=['csv'])
    # if datafile is not None:
    #   file_details = {"FileName": datafile.name, "FileType": datafile.type}
    #  df = pd.read_csv(datafile)
    # st.dataframe(df)
    # save_uploadedFile(datafile)

    # def file_Selector(folder_path='./datasets'):
    #     filenames = os.listdir(folder_path)
    #     selected_filename = st.selectbox('Select A file', filenames)
    #     return os.path.join(folder_path, selected_filename)

    filename = datafile
    st.info('You selected {}'.format(filename))

    if filename is not None:
        # read data
        df = pd.read_csv(filename)

    # show dataset
    if filename is not None and st.checkbox("Show Dataset"):
        number = int(st.number_input("Number of rows to view", None, None, 1))
        st.dataframe(df.head(number))

    # show columns
    if filename is not None and st.button("View Column Names"):
        st.write(df.columns)

    # show shapes
    if filename is not None and st.checkbox("Show Shape of the dataset"):
        st.write(df.shape)
        dataDim = st.radio("Show Dimension by ", ("Rows", "Columns"))
        if dataDim == 'Rows':
            st.text("Number of Rows")
            st.write(df.shape[0])
        elif dataDim == 'Columns':
            st.text("Number of Rows")
            st.write(df.shape[1])
        else:
            st.write(df.shape)

    # select columns
    if filename is not None and st.checkbox("Select columns to show"):
        all_columns = df.columns.tolist()
        selected_columns = st.multiselect("Select", all_columns)
        new_df = df[selected_columns]
        st.dataframe(new_df)

    # show values
    if filename is not None and st.button("Value Counts"):
        st.text("Value Counts by Target/Class")
        st.write(df.iloc[:, -1].value_counts())

    # if st.button("Data Types"):
    #     st.write(df.dtypes)

    # show summary
    if filename is not None and st.checkbox("Summary"):
        st.write(df.describe().T)

    # plot visualization

    # correlation plot

    # Seaborn plot

    # count plot

    # pie chart

    # customizable plot
    st.subheader("Data Visualization")
    # Correlation
    # Seaborn Plot
    if filename is not None and st.checkbox("Correlation Plot[Seaborn]"):
        st.write(sns.heatmap(df.corr(), annot=True))
        st.pyplot()

        # Pie Chart
    if filename is not None and st.checkbox("Pie Plot"):
        all_columns_names = df.columns.tolist()
        if st.button("Generate Pie Plot"):
            st.success("Generating A Pie Plot")
            st.write(df.iloc[:, -1].value_counts().plot.pie(autopct="%1.1f%%"))
            st.pyplot()

        # Count Plot
    if filename is not None and st.checkbox("Plot of Value Counts"):
        st.text("Value Counts By Target")
        all_columns_names = df.columns.tolist()
        primary_col = st.selectbox(
            "Primary Columm to GroupBy", all_columns_names)
        selected_columns_names = st.multiselect(
            "Select Columns", all_columns_names)
        if st.button("Plot"):
            st.text("Generate Plot")
            if selected_columns_names:
                vc_plot = df.groupby(primary_col)[
                    selected_columns_names].count()
            else:
                vc_plot = df.iloc[:, -1].value_counts()
            st.write(vc_plot.plot(kind="bar"))
            st.pyplot()

        # Customizable Plot
        if filename is not None:
            st.subheader("Custom Data Visualization")
            all_columns_names = df.columns.tolist()
            type_of_plot = st.selectbox("Select Type of Plot", [
                "area", "bar", "line", "hist", "box", "kde"])
            selected_columns_names = st.multiselect(
                "Select Columns To Plot", all_columns_names)

        if filename is not None and st.button("Generate Plot"):
            st.success("Generating Customizable Plot of {} for {}".format(
                type_of_plot, selected_columns_names))

        # Plot By Streamlit
        if type_of_plot == 'area':
            cust_data = df[selected_columns_names]
            st.area_chart(cust_data)

        elif type_of_plot == 'bar':
            cust_data = df[selected_columns_names]
            st.bar_chart(cust_data)

        elif type_of_plot == 'line':
            cust_data = df[selected_columns_names]
            st.line_chart(cust_data)

        # Custom Plot
        elif type_of_plot:
            cust_plot = df[selected_columns_names].plot(kind=type_of_plot)
            st.write(cust_plot)
            st.pyplot()


if __name__ == "__main__":
    main()
