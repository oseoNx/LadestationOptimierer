#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 13:50:45 2022

@author: alex
"""

import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px
import pickle
import numpy as np
import sklearn
from shapely import wkt


@st.cache(allow_output_mutation=True)
def load_data():
    data = pd.read_csv('./Data/final_data.csv', index_col='Gemeindename')
    return data


@st.cache(allow_output_mutation=True)
def load_borders():
    borders = pd.read_csv("./Data/GemeindeGrenzen_Vereinfacht.csv", index_col='GemeindeLabel')
    borders['geometry'] = borders['geometry'].apply(wkt.loads)
    return gpd.GeoDataFrame(borders)

@st.cache(allow_output_mutation=True)
def load_EV():
    EV = pd.read_csv('./Data/aggrigated_data.csv', index_col='BFS-Nr')
    return EV[['Gemeindename','EV_Bestand_2010', 'EV_Bestand_2011', 'EV_Bestand_2012', 'EV_Bestand_2013', 'EV_Bestand_2014', 'EV_Bestand_2015', 'EV_Bestand_2016', 'EV_Bestand_2017', 'EV_Bestand_2018', 'EV_Bestand_2019', 'EV_Bestand_2020', 'EV_Bestand_2021']]

@st.cache()
def load_stations():
    stations = pd.read_csv('./Data/Chargingstations_melted.csv')
    return stations

@st.cache()
def load_model():
    loaded_model = pickle.load(open('./Data/finalized_model.sav', 'rb'))
    return loaded_model

def find_center(GemName, gdf):
    return gdf[gdf.index.isin(GemName)].geometry.values[0].centroid

def df_growth(df, ev_growth, pop_growth, sector_3_growth):
    print('f start')
    df_estm = df.copy()
    print('copy')
    df_estm['EV_Bestand_2021'] = np.floor(df_estm['EV_Bestand_2021'] * (1+ev_growth))
    df_estm['Anz_Einwohner'] = np.floor(df_estm['Anz_Einwohner'] * (1+pop_growth))
    df_estm['Beschäftigte_3_Sektor'] = np.floor(df_estm['Beschäftigte_3_Sektor'] * (1+sector_3_growth))
    print('wachstum')
    df_estm['Ladestationen_optimiert'] = load_model().predict(df_estm.drop(columns=['Ladestationen_optimiert','aktl_Ladestationen','EU_Anforderung','EU Differenz','BFS-Nr','Differenz'],axis=1))
    print('model')
    df_estm['EU_Anforderung'] = df_estm['EV_Bestand_2021'] / 10 
    df_estm['EU_Anforderung'] = df_estm(lambda x: int(x['EU_Anforderung']), axis=1)
    print('eu')
    return df_estm

    
st.set_page_config(
    page_title= "Ladestation Optimierer",
    page_icon = "🔋",
    layout="wide"
    )
st.title('Ladestationen Schweiz Übersicht')

df = load_data()
borders = load_borders()
EVdf = load_EV()
stations = load_stations()

df['EU_Anforderung'] = df['EV_Bestand_2021'] / 10 
df['EU_Anforderung'] = df.apply(lambda x: int(x['EU_Anforderung']), axis=1)
df['EU Differenz'] = df['aktl_Ladestationen'] - df['EU_Anforderung']


tab1, tab2, tab3 = st.tabs(["Analyse nach Gemeinde", "Analyse Schweiz","Wachstumsrechner"])

with tab1:


    options1 = st.multiselect(
        'Geben Sie eine Gemeinde ein',
        df.index,
        ['Zürich'],
        key=1)
       # max_selections = 1)
    try:
        loc =df[df.index.isin(options1)]
        loc2=EVdf[EVdf.Gemeindename.isin(options1)]
        
        
        row1_col1, row1_col2, row1_col3, row1_col4, row1_col5, row1_col6 = st.columns([2.5,2.5,2,2.5,2.5,2.5])
        row2_col1, row2_col2, row2_col3, row2_col4, row2_col5, row2_col6 = st.columns([2.5,2.5,2.5,2.5,2.5,2.5])
        
        row2_col2.metric("Optimale Anz. Ladestationen", str(int(loc['Ladestationen_optimiert'].values[0])), str( int(0-loc['Differenz'].values[0].round(0))),delta_color="off", help='Das Delta zeigt die Differenz zur aktuellen Anz. Ladestation an.' )
        print(1)
        row2_col4.metric("Einwohner Anz.", str(int(loc['Anz_Einwohner'].values[0])))
        print(2)
        row2_col6.metric("Anz. EV Bestand 2021", str(int(loc['EV_Bestand_2021'].values[0])), str(int((loc['EV_Bestand_2021'].values[0] - loc2['EV_Bestand_2020'].values[0]))),help='Das Delta zeigt die Differenz zum Bestand EV 2020 an.')
        print(3)
        row1_col1.metric("Akutelle Anz. Ladestationen", str(int(loc['aktl_Ladestationen'].values[0])))
        print(4)
        row1_col5.metric("Arbeitende im 3. Sektor", str(int(loc['Beschäftigte_3_Sektor'].values[0])))
        print(5)
        #row2_col4.metric("Strassenlänge (Km)", str(int(loc['Strassenlänge(km)'].values[0])))
        print(6)
        row1_col3.metric("EU Anfforderung", str(int(loc['EU_Anforderung'].values[0])), str( int(loc['EU Differenz'].values[0])), help='Gemäss EU Anfforderungen müssen pro 10 EV einen öffentlichen Ladepunkt gewährleistet werden. Das Delta zeigt die Differenz zur Anfforderung auf.')
        print(7)
    
        select = st.selectbox(
        'Bitte wählen Sie einen Kartenfilter aus',
        df.drop(['BFS-Nr','Gemeinde_Kategorie'],axis=1).columns, index=24, key=2)
        
        fig1 = px.choropleth_mapbox(df,
                               geojson=borders.geometry,
                               locations=df.index,
                               color=select,
                               center={"lat": find_center(options1, borders).y, "lon": find_center(options1, borders).x},
                               mapbox_style="open-street-map",
                               zoom=9,
                               color_continuous_scale="rdbu", range_color=[df[select].min()*0.2,df[select].max()*0.2],
                               opacity=0.5, height=800)
    
    
        st.plotly_chart(fig1, use_container_width=True)
    except:
        st.warning('Bitte geben Sie genau eine Gemeinde ein.')
                 #  icon="⚠️")


with tab2:
    
    select2 = st.selectbox(
    'Bitte wählen Sie einen Kartenfilter aus',
    df.drop(['BFS-Nr','Gemeinde_Kategorie'],axis=1).columns, index=24, key=3)

    fig2 = px.choropleth_mapbox(df,
                           geojson=borders.geometry,
                           locations=df.index,
                           color=select2,
                           center={"lat": 46.9, "lon": 8.2275},
                           mapbox_style="open-street-map",
                           zoom=7,
                           color_continuous_scale="rdbu", range_color=[df[select2].min()*0.2,df[select2].max()*0.2],
                           opacity=0.5, height=800)


    st.plotly_chart(fig2, use_container_width=True)
    
    row3_col1, row3_col2 = st.columns([5,5])
    
    row3_col1.subheader("Elektroautobestand nach Gemeinde")
    options2 = row3_col1.multiselect(
        'Welche Gemeinde wollen Sie vergleichen?',
        EVdf['Gemeindename'],
        ['Zürich', 'Basel', 'Bern', 'St. Gallen'],key=4)
    
    melt_df = EVdf.melt(id_vars=['Gemeindename'],var_name='year',value_name='anzahl EV')
    melt_df['year'] = melt_df.apply(lambda x: x['year'].replace('EV_Bestand_',''),axis=1)
    filtered_ev = melt_df[melt_df['Gemeindename'].isin(options2)]
    
    fig3 = px.line(filtered_ev, x="year", y="anzahl EV", color='Gemeindename')
    fig3.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    
    
    row3_col2.subheader("Anz. Ladestationen nach Kanton")
    options3 = row3_col2.multiselect(
        'Welche Kantone wollen Sie vergleichen?',
        pd.unique(stations['Canton']),
        ['ZH', 'SG', 'AG', 'BS'],
        key=5)
    
    filtered_stations = stations[stations['Canton'].isin(options3)]
    
    fig4 = px.line(filtered_stations, x="datetime", y="Amount", color='Canton')
    fig4.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    row3_col1.plotly_chart(fig3, use_container_width=True)
    row3_col2.plotly_chart(fig4, use_container_width=True)
    
    st.subheader("Gemeinde Vergleich nach Feature")
    row4_col1, row4_col2 = st.columns([5,5])
    
    
    options3 = row4_col2.multiselect(
        'Welche Gemeinde wollen Sie vergleichen?',
        df.index,
        ['Zürich', 'Basel', 'Bern', 'St. Gallen'],
        key=6)
    
    select3 = row4_col2.selectbox(
    'Bitte wählen Sie einen Kartenfilter aus',
    df.drop(['BFS-Nr','Gemeinde_Kategorie'],axis=1).columns, index=1, key=7)
    
    filtered_data = df[df.index.isin(options3)]
    
    fig5 = px.bar(filtered_data, x=filtered_data.index, y=select3)
    row4_col1.plotly_chart(fig5)
    
    st.subheader("Übersicht Datenset")
    
    st.dataframe(df, height= 800)

    
    
with tab3:

    st.subheader("Gemeindewachstumsrechner")

    options4 = st.multiselect(
        'Geben Sie eine Gemeinde ein',
        df.index,
        ['Zürich'],
        key=8)
       # max_selections = 1)
    row1_col1, row1_col2, row1_col3 = st.columns([2.5,2.5,2.5])
    przEinw = row1_col1.slider('Geben Sie bitte die Einwohnerwachstumsrate an.', 0.0, 0.5, 0.007, key=9)
    prz3Sek = row1_col2.slider('Geben Sie bitte die Wachstumsrate der Arbeitende im 3. Sektor an.', 0.0, 1.0, 0.05, key=10)
    przEV = row1_col3.slider('Geben Sie bitte die EV-Wachstumsrate an.', 0.0, 2.0, 0.6, key=11)
    
    
    try:
        doc =df[df.index.isin(options4)]
        doc2=EVdf[EVdf.Gemeindename.isin(options4)]
        print('a')
        doc3 = df_growth(doc, przEV, przEinw, prz3Sek)
        print('b')
        row2_col1, row2_col2, row2_col3, row2_col4, row2_col5, row2_col6 = st.columns([2.5,2.5,2,2.5,2.5,2.5])
        row3_col1, row3_col2, row3_col3, row3_col4, row3_col5, row3_col6 = st.columns([2.5,2.5,2.5,2.5,2.5,2.5])

        ### Darstellung 2021 ###
        row2_col1.metric("Optimale Anz. Ladestationen ALT", str(int(doc['Ladestationen_optimiert'].values[0])), str( int(0-doc['Differenz'].values[0].round(0))),delta_color="off", help='Das Delta zeigt die Differenz zur aktuellen Anz. Ladestation an.' )
        print(1)
        row2_col4.metric("Einwohner Anz. ALT", str(int(doc['Anz_Einwohner'].values[0])))
        print(2)
        row2_col6.metric("Anz. EV Bestand ALT", str(int(doc['EV_Bestand_2021'].values[0])), str(int((doc['EV_Bestand_2021'].values[0] - doc2['EV_Bestand_2021'].values[0]))),help='Das Delta zeigt die Differenz zum Bestand EV 2020 an.')
        print(3)
        row2_col1.metric("Akutelle Anz. Ladestationen ALT", str(int(doc['aktl_Ladestationen'].values[0])))
        print(4)
        row2_col5.metric("Arbeitende im 3. Sektor ALT", str(int(doc['Beschäftigte_3_Sektor'].values[0])))
        print(5)
        #row3_col4.metric("Strassenlänge (Km)", str(int(doc['Strassenlänge(km)'].values[0])))
        print(6)
        row2_col3.metric("EU Anfforderung ALT", str(int(doc['EU_Anforderung'].values[0])), str( int(doc['EU Differenz'].values[0])), help='Gemäss EU Anfforderungen müssen pro 10 EV einen öffentlichen Ladepunkt gewährleistet werden. Das Delta zeigt die Differenz zur Anfforderung auf.')
        print(7)
        

        ### Darstellung mit Wachstumsrate ###
        row3_col1.metric("Optimale Anz. Ladestationen NEU", str(int(doc3['Ladestationen_optimiert'].values[0])), str( int(0-doc3['Differenz'].values[0].round(0))),delta_color="off", help='Das Delta zeigt die Differenz zur aktuellen Anz. Ladestation an.' )
        print(8)
        row3_col4.metric("Einwohner Anz. NEU", str(int(doc3['Anz_Einwohner'].values[0])))
        print(9)
        row3_col6.metric("Anz. EV Bestand NEU", str(int(doc3['EV_Bestand_2021'].values[0])), str(int((doc3['EV_Bestand_2021'].values[0] - doc2['EV_Bestand_2021'].values[0]))),help='Das Delta zeigt die Differenz zum Bestand EV 2020 an.')
        print(10)
        row3_col1.metric("Akutelle Anz. Ladestationen NEU", str(int(doc3['aktl_Ladestationen'].values[0])))
        print(11)
        row3_col5.metric("Arbeitende im 3. Sektor NEU", str(int(doc3['Beschäftigte_3_Sektor'].values[0])))
        print(12)
        #row3_col4.metric("Strassenlänge (Km)", str(int(doc['Strassenlänge(km)'].values[0])))
        print(13)
        row3_col3.metric("EU Anfforderung NEU", str(int(doc['EU_Anforderung'].values[0])), str( int(doc3['EU Differenz'].values[0])), help='Gemäss EU Anfforderungen müssen pro 10 EV einen öffentlichen Ladepunkt gewährleistet werden. Das Delta zeigt die Differenz zur Anfforderung auf.')
        print(14)
    
    
    except:
        st.warning('Bitte geben Sie genau eine Gemeinde ein.')
                 #  icon="⚠️")   
    
    
    
    
    
    
    
    
    
