import streamlit as st

#import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px
import plotly.graph_objects as go

import pandas as pd
import numpy as np
from scipy import stats

#################### Install Components

from streamlit_option_menu import option_menu

############### 

st.set_page_config(
    page_title="Linear Regression Fitting app",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

st.image("logo_url.png", width=100)##### I need a logo here

################################################################
# place for components


################################################################

efermi = 5.0
epsilon = np.linspace(0.0,30,8000)
epsilon_laburtu = np.linspace(-30.0,0.0,8000)
t_mu = np.linspace(0.0,11,500)

def muT(efermi,t_mu):
    Tmu = t_mu
    return efermi*(1.0 - (Tmu/efermi)**2.00)

def f_FD_kittel(epsilon, tenperatura, efermi):
    energy = epsilon
    T = tenperatura
    return 1.0/(np.exp((energy-(efermi*(1.0 - (T/efermi)**2.00)))*1./T) + 1.0)

temperatures = pd.DataFrame([0.00000, 0.05, 0.50, 1.00, 2.50, 5.00, 10.00],columns=['T'], dtype = float) # hauexek dira erabili beharreko tenperaturak
temperatures['colors'] = ['#636EFA','#EF553B','#00CC96','#AB63FA','#FFA15A','#19D3F3','#FF6692']#,'#B6E880','#FF97FF','#FECB52']
temperatures['mu'] = muT(efermi,temperatures['T'])
temperatures['mu_05'] = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
temperatures['mu_00'] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


col0,col00 = st.columns((3,10))
with col0:
   tenperaturak = st.select_slider(label=r'$\textrm{Tenperaturak } /10^{5}; \quad\epsilon_{F} = 50.000 \rm{~K}$', options=temperatures['T'], value=temperatures['T'][1])

col1, padding, col2 = st.columns((10,2,10))
col3, padding, col4 = st.columns((10,2,10))

f_FD_kittel_funtzioa = f_FD_kittel(epsilon, tenperaturak, efermi)
f_FD_kittel_funtzioa_laburtu = f_FD_kittel(epsilon_laburtu, tenperaturak, efermi)
muT_funtzioa = muT(efermi,t_mu)

fig_banaketa = px.line(x=epsilon, y=f_FD_kittel_funtzioa, range_x=[-0.50,10.0], range_y=[-0.15,1.15], title="FD banaketa-funtzioa", width=700, height=400)
fig_banaketa.add_traces(go.Scatter(x=[muT(efermi,tenperaturak)], y=[0.5], showlegend=False, marker=dict(symbol='circle',size=10,color=temperatures['colors'],),))
fig_banaketa.add_traces(go.Scatter(x=[0.0,5.], y=[1.0,1.0], mode ="lines", line_color='orange', line_width=3, opacity=0.5, showlegend=False,))
fig_banaketa.add_traces(go.Scatter(x=[5.0,10.0], y=[0.0,0.0], mode ="lines", line_color='orange', line_width=3, opacity=0.5, showlegend=False,))
fig_banaketa.add_traces(go.Scatter(x=[5.,5.], y=[0.,1.], line_color='orange', mode ="lines", line_width=3, opacity=0.5, showlegend=False,))

fig_potentziala = px.line(x=t_mu, y=muT_funtzioa, range_x=[-0.50,11.0], range_y=[-20.25,6.00], title="Potentzial kimikoaren T-rekiko eboluzioa", width=700, height=400)
fig_potentziala.add_traces(go.Scatter(x=[tenperaturak], y=[muT(efermi,tenperaturak)], showlegend=False, marker=dict(symbol='circle',size=10,color=temperatures['colors'],),))

fig_banaketa_laburtu = px.line(x=epsilon_laburtu-muT(efermi,tenperaturak), y=f_FD_kittel_funtzioa_laburtu, range_x=[-10.00,10.0], range_y=[-0.15,1.15], title="FD banaketa-funtzioa", width=600, height=400)
fig_banaketa_laburtu.add_scatter(x=epsilon-muT(efermi,tenperaturak), y=f_FD_kittel_funtzioa, line={'color': '#636EFA'}, showlegend=False)
fig_banaketa_laburtu.add_traces(go.Scatter(x=[-10.,0.], y=[1.0,1.0], mode ="lines", line_color='orange', line_width=3, opacity=0.5, showlegend=False,))
fig_banaketa_laburtu.add_traces(go.Scatter(x=[0.0,10.0], y=[0.0,0.0], mode ="lines", line_color='orange', line_width=3, opacity=0.5, showlegend=False,))
fig_banaketa_laburtu.add_traces(go.Scatter(x=[0,0], y=[0,1], line_color='orange', mode ="lines", line_width=3, opacity=0.5, showlegend=False,))
fig_banaketa_laburtu.add_traces(go.Scatter(x=[0.0], y=[0.5], marker=dict(symbol='circle', size=10, color= temperatures['colors']), showlegend=False))

fig_banaketa.add_traces(go.Scatter(x=temperatures['mu'],y=temperatures['mu_05'], marker=dict(symbol='circle', size=10, color=temperatures['colors']) ))#, showlegend=False))
fig_potentziala.add_traces(go.Scatter(x=temperatures['mu_00'],y=temperatures['mu'], marker=dict(symbol='circle', size=10, color=temperatures['colors']) ))#, showlegend=False))

col1.write(fig_banaketa)
col2.write(fig_potentziala)
col3.write(fig_banaketa_laburtu)

#st.components.v1.html(fig.to_html(include_mathjax='cdn'),)

#st.plotly_chart(fig_banaketa, theme='streamlit', use_container_width=False)