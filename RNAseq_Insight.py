from click import progressbar
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from sanbomics.tools import id_map
from sanbomics.plots import volcano
import scanpy as sc
import time
import requests
import gseapy
from gseapy.plot import barplot
from streamlit_lottie import st_lottie

import sys  # Ensure sys module is imported

st.set_page_config("RNAseq Analyzer")

with open('style.css') as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def load_lottieurl(url:str):
    r=requests.get(url)
    if r.status_code != 200 :
        return None
    return r.json()

def data_description(df):
    # Define the number of columns
    num_columns = 2
    # Calculate the number of sections
    num_sections = 4
    # Calculate the number of rows needed
    num_rows = (num_sections + num_columns - 1) // num_columns

    # Loop over each row
    for i in range(num_rows):
        cols = st.columns(num_columns)
        # Loop over each column in the row
        for j in range(num_columns):
            index = i * num_columns + j
            if index < num_sections:
                with cols[j]:
                    if index == 0:
                        st.write("Summary Statistics:")
                        summary_stats = df.describe().T
                        st.dataframe(summary_stats)
                    elif index == 1:
                        st.write("Data Types")
                        data_types = pd.DataFrame(df.dtypes, columns=['Data Type'])
                        data_types.reset_index(inplace=True)
                        data_types.columns = ['Column Name', 'Data Type']
                        st.dataframe(data_types)
                    elif index == 2:
                        st.write("Missing Values")
                        missing_values = pd.DataFrame(df.isnull().sum(), columns=['Missing Values'])
                        missing_values.reset_index(inplace=True)
                        missing_values.columns = ['Column Name', 'Missing Values']
                        st.dataframe(missing_values)
                    elif index == 3:
                        st.write("Unique Values")
                        unique_values = pd.DataFrame(df.nunique(), columns=['Unique Values'])
                        unique_values.reset_index(inplace=True)
                        unique_values.columns = ['Column Name', 'Unique Values']
                        st.dataframe(unique_values)

def data_filteration(df):
    counts = df.set_index(df.columns[0])
    counts = counts[counts.sum(axis=1) > 0].T

    # Extract first two characters from each index
    groups = counts.groupby(counts.index.str[:2]).sum()
    # Create a DataFrame with 'Sample' and 'Condition' columns
    metadata_df = pd.DataFrame({'Sample': counts.index, 'Condition': counts.index.str[:2]})
    
    return counts, metadata_df

def create_deseq_dataset(counts, metadata_df):
    st.write("DESeq2 Analysis")
    progress_bar = st.progress(0)

    # Ensure proper alignment of indices
    counts = counts.reset_index(drop=True)
    metadata_df = metadata_df.reset_index(drop=True)

    # Create DESeq dataset
    dds = DeseqDataSet(counts=counts.astype(int), metadata=metadata_df, design_factors='Condition')
    dds.deseq2()
    c = metadata_df['Condition'].unique()

    # Perform DESeq2 analysis for each pair of conditions
    total_steps = len(c) * (len(c) - 1) // 2
    step = 0
    for i in range(len(c)):
        for j in range(i + 1, len(c)):
            con_one = c[i]
            con_two = c[j]

            st.subheader(f"DESeq2 Results for Conditions: {con_one} vs {con_two}")
            stat_res = DeseqStats(dds, contrast=('Condition', con_one, con_two))
            stat_res.summary()
            res = stat_res.results_df

            # Filter results based on baseMean >= 10
            res = res[res.baseMean >= 10]

            st.write(res)

            # Add symbol column
            add_symbols(res)

            # Visualize results
            visualize_results(res)
            pca_plot = pca(res)
            if pca_plot:
                st.pyplot(pca_plot)
            Gene_Pathway(res)

            step += 1
            progress_bar.progress(step / total_steps)
    
    # Clear the progress bar after completion
    progress_bar.empty()

def add_symbols(res):
    species_opt = st.radio('Select species:', ['Human', 'Mice'])
    if species_opt == 'Human':
        mapper = id_map(species='human')
    elif species_opt == "Mice":
        mapper = id_map(species='mice')
    res['Symbol'] = res.index.map(mapper.mapper)
    st.write(res)

def visualize_results(res):
    st.subheader("Visualization")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter plot of all genes
    ax.scatter(res['log2FoldChange'], -np.log10(res['pvalue']), alpha=0.5, label='All Genes', color='blue')

    # Scatter plot of significant genes
    sig_genes = res[res['padj'] < 0.05]
    ax.scatter(sig_genes['log2FoldChange'], -np.log10(sig_genes['pvalue']), color='red', s=100, label='Significant Genes')

    # Set labels and title
    ax.set_xlabel('log2 Fold Change', fontsize=14)
    ax.set_ylabel('-log10 (p-value)', fontsize=14)
    ax.set_title('Volcano Plot', fontsize=16)

    # Add grid lines
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Annotate significant genes
    for i, symbol in enumerate(sig_genes['Symbol']):
        ax.annotate(symbol, (sig_genes['log2FoldChange'].iloc[i], -np.log10(sig_genes['pvalue'].iloc[i])),
                    fontsize=10, ha='center', va='bottom')

    # Add legend
    ax.legend(fontsize=12)

    # Show the plot in Streamlit
    st.pyplot(fig)

    res['A'] = (res['baseMean'] + 1).transform(np.log2)

    # Create the MA plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(res['A'], res['log2FoldChange'], alpha=0.5)

    sig_genes = res[res['padj'] < 0.05]
    ax.scatter(sig_genes['A'], sig_genes['log2FoldChange'], color='red', s=100, label='Significant Genes')

    # Add labels and title
    ax.set_xlabel('Average Expression (A)', fontsize=14)
    ax.set_ylabel('log2 Fold Change (M)', fontsize=14)
    ax.set_title('MA Plot', fontsize=16)

    # Add a horizontal line at M=0 for reference
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)

    # Add grid lines
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Add legend
    ax.legend(fontsize=12)

    # Show the plot in Streamlit
    st.pyplot(fig)

def pca(res):
    try:
        # Perform PCA
        res = res.dropna()
        pca = PCA(n_components=2)
        pca_results = pca.fit_transform(res[['baseMean', 'log2FoldChange', 'lfcSE', 'stat', 'pvalue', 'padj']])

        # Create the PCA plot
        fig, ax = plt.subplots(figsize=(8, 6))

        # Scatter plot of PCA results
        scatter = ax.scatter(pca_results[:, 0], pca_results[:, 1], alpha=0.5, label='Data Points')

        # Add labels and title
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title('PCA Plot')

        # Add legend for explained variance ratio
        ax.legend(*scatter.legend_elements(), title="Explained Variance Ratio")

        # Add grid lines
        ax.grid(True, linestyle='--', linewidth=0.5)

        return fig  # Return the Matplotlib figure object
    except Exception as e:
        st.error(f"An error occurred during PCA analysis: {e}")
        return None

def Gene_Pathway(res):
    all_symbols_list = res['Symbol'].astype(str).unique().tolist()
    res_enrichr = gseapy.enrichr(gene_list=all_symbols_list,
                                  gene_sets='GO_Biological_Process_2021',
                                  cutoff=0.05)

    st.write(res_enrichr.results)
    barplot(res_enrichr.results, top_term=10, figsize=(8, 6), title='Enrichment Analysis',
            save='barplot.png')

    all_symbols_list = res['Symbol'].unique().tolist()
    res_enrichr2 = gseapy.enrichr(gene_list=all_symbols_list,
                                   gene_sets='GO_Biological_Process_2021',
                                   cutoff=0.05)

    st.write(res_enrichr2.results)
    barplot(res_enrichr2.results, top_term=10, figsize=(8, 6), title='Enrichment Analysis',
            save='barplot.png')

    all_symbols_list = res['Symbol'].astype(str).unique().tolist()
    res = gseapy.enrichr(gene_list=all_symbols_list,
                    gene_sets='GO_Biological_Process_2021',
                    cutoff=0.05)

    st.write(res.results)
    barplot(res.results, top_term=10, figsize=(8, 6), title='Enrichment Analysis',
        save='barplot.png')

    all_symbols_list = res['Symbol'].unique().tolist()
    res = gseapy.enrichr(gene_list=all_symbols_list,
                    gene_sets='GO_Biological_Process_2021',
                    cutoff=0.05)

    st.write(res.results)
    barplot(res.results, top_term=10, figsize=(8, 6), title='Enrichment Analysis',
        save='barplot.png')


def display_guide():
    st.title("Welcome to SSBPRNAseq Analyzer")
    st.subheader("Guide for Using the App")
    st.write("1. Upload your RNA-seq data file in CSV or TXT format using the file uploader on the left sidebar.")
    st.write("2. After uploading the file, choose an analysis option from the sidebar to perform different analyses on your data.")
    st.write("3. Follow the instructions provided under each analysis option to proceed with the analysis.")
    st.write("4. Once the analysis is completed, you will see the results displayed on the main panel.")

def main():
    # st.title("*PyDESeq2 Analysis* \U0001F9EC")

    uploaded_file = st.sidebar.file_uploader("Choose a file", type=["csv", "txt"])
    
    if uploaded_file is None:
        # Display Lottie animation if no file is uploaded
        lottie_logo = load_lottieurl(url="https://lottie.host/7397f866-d481-4aae-9c48-69c5345f4d65/F9EyPLtJzM.json")
        st_lottie(lottie_logo, key='hello')
        # st.warning("Please upload a file.")
        return

    progress_bar = st.sidebar.progress(0)
    progress_text = st.sidebar.empty()
    progress_text.text("Uploading... 0%")
        
    # Simulate a delay for file uploading
    for percent_complete in range(100):
        time.sleep(0.1)  # Adjust the sleep time as needed
        progress_bar.progress(percent_complete + 1)
        progress_text.text(f"Progressing.. {percent_complete+1}%")
        
    opt_unicode = {
        'Guide': '📚',
        'Data Description': "\U0001F4CA",  # Unicode for file folder
        'Data Filtration': "\U0001F50E",   # Unicode for magnifying glass
        'Create DESeq dataset': "\U0001F4DD"  # Unicode for notebook
        
    }
    st.sidebar.divider()
    opt = st.sidebar.radio('Select analysis', list(opt_unicode.keys()), format_func=lambda x: f"{opt_unicode[x]} {x}")
    st.sidebar.divider()
        
    data = pd.read_csv(uploaded_file)   
    if opt == 'Guide':
        st.write(display_guide())       
    if opt == 'Data Description':
        data_description(data)
    elif opt == 'Data Filtration':
        counts, metadata_df = data_filteration(data)
        st.write("Counts:")
        st.write(counts)
        st.write("Metadata:")
        st.write(metadata_df)
    elif opt == 'Create DESeq dataset':
        counts, metadata_df = data_filteration(data)
        create_deseq_dataset(counts, metadata_df)

    # Clear the progress bar and text
    progress_bar.empty()
    progress_text.empty()

if __name__ == "__main__":
    main()
