import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit

st.title("E. coli Growth Analysis Web App")
st.write("Upload an Excel file to analyze OD600 growth curves.")

# File upload
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file is not None:
    # Get sheet names
    excel_file = pd.ExcelFile(uploaded_file)
    sheet_names = excel_file.sheet_names

    # Button-like selection for sheet choice
    selected_sheet = st.radio("Select a sheet", sheet_names)

    # Analyze button
    if st.button("Analyze Growth Curves"):
        with st.spinner("Analyzing growth curves..."):
            try:
                # Load data from the selected sheet
                df_growth = pd.read_excel(uploaded_file, sheet_name=selected_sheet)

                # Convert 'Time' column to hours
                df_growth['Time_hours'] = df_growth['Time'].apply(lambda x: x.hour + x.minute / 60 + x.second / 3600)

                # Define OD columns groups
                od_columns_list = [
                    ['B9', 'C9', 'D9'], 
                    ['B10', 'C10', 'D10'], 
                    ['B11', 'C11', 'D11'], 
                    ['E9', 'F9', 'G9'], 
                    ['E10', 'F10', 'G10'], 
                    ['E11', 'F11', 'G11']
                ]

                colors = ['rgb(145,179,222)', 'rgb(145,222,212)', 'rgb(153,255,153)', 
                          'rgb(255,232,154)', 'rgb(224,129,228)', 'rgb(255,154,201)']

                subplot_titles = ['WT-LB', '1+2-LB_A+K', '2+4-LB_A+K', 
                                  '2+3-LB_A', '1+2-LB_A+K+Ara', '2+4-LB_A+K+Ara']

                # Define exponential model
                def exponential_model(t, a, b):
                    return a * np.exp(b * t)

                # Create subplots
                fig = make_subplots(rows=2, cols=3, shared_xaxes=True, vertical_spacing=0.05, subplot_titles=subplot_titles)

                for idx, od_columns in enumerate(od_columns_list):
                    df_growth['OD_mean'] = df_growth[od_columns].mean(axis=1)
                    df_growth['OD_std'] = df_growth[od_columns].std(axis=1)

                    best_r2 = -np.inf
                    best_fit_params = None
                    best_time_range = None

                    n_points = len(df_growth[od_columns[0]])
                    for start in range(0, n_points - 16):
                        for end in range(start + 16, n_points):
                            time_slice = df_growth['Time_hours'][start:end]
                            od_slice = df_growth['OD_mean'][start:end]

                            try:
                                popt, _ = curve_fit(exponential_model, time_slice, od_slice, maxfev=5000)
                                od_pred = exponential_model(time_slice, *popt)
                                r2 = r2_score(od_slice, od_pred)

                                if r2 > best_r2:
                                    best_r2 = r2
                                    best_fit_params = popt
                                    best_time_range = (time_slice.iloc[0], time_slice.iloc[-1])

                            except RuntimeError:
                                continue

                    if best_fit_params is not None:  # Check if best_fit_params is not None
                        a_best, b_best = best_fit_params
                        t_start, t_end = best_time_range
                        generation_time = np.log(2) / b_best

                        fig.add_trace(go.Scatter(
                            x=df_growth['Time_hours'], y=df_growth['OD_mean'], mode='markers',
                            error_y=dict(type='data', array=df_growth['OD_std'], visible=True),
                            marker=dict(color=colors[idx])
                        ), row=(idx // 3) + 1, col=(idx % 3) + 1)

                        time_fit = np.linspace(t_start, t_end, 100)
                        od_fit = exponential_model(time_fit, a_best, b_best)
                        fig.add_trace(go.Scatter(
                            x=time_fit, y=od_fit, mode='lines', line=dict(color='red')
                        ), row=(idx // 3) + 1, col=(idx % 3) + 1)

                        fig.add_annotation(
                            x=8.3, y=0.2, xref='x', yref='y',
                            text=f"OD600 = {a_best:.4f} * exp({b_best:.4f} * t)<br>R² = {best_r2:.4f}<br>Gen Time: {generation_time:.2f}h",
                            showarrow=False, font=dict(size=12, color='blue'),
                            row=(idx // 3) + 1, col=(idx % 3) + 1
                        )

                fig.update_xaxes(title_text='Time (hours)', range=[0, 11],linecolor='black', linewidth=1, showline=True, mirror=True,)
                fig.update_yaxes(title_text='OD600',linecolor='black', linewidth=1, showline=True, mirror=True,)

                fig.update_layout(
                    title_text="E. coli Growth Curve Analysis",
                    height=800, width=1500, plot_bgcolor='white', showlegend=False, title_x=0.5
                )

                st.plotly_chart(fig)

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

