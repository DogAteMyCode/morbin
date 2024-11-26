import locale
locale.setlocale(locale.LC_TIME, 'es_MX')
from dash import Dash, html, dcc, callback, Output, Input
from sklearn.preprocessing import StandardScaler
from plotly.graph_objs import Scatter
from sklearn.decomposition import PCA
import plotly.express as px
import geopandas as gpd
import pandas as pd

data = pd.read_csv('new.zip', parse_dates=['date'])

########


data_filtered = data[~data['state'].str.contains("total global", case=False)]

data_pivot_filtered = data_filtered.pivot_table(index='state', columns='sickness', values='value', aggfunc='sum', fill_value=0)

scaler = StandardScaler()
scaled_data_filtered = scaler.fit_transform(data_pivot_filtered)

pca_filtered = PCA(n_components=2)
pca_result_filtered = pca_filtered.fit_transform(scaled_data_filtered)

pca_df_filtered = pd.DataFrame(pca_result_filtered, columns=['PC1', 'PC2'])
pca_df_filtered['state'] = data_pivot_filtered.index

fig_filtered = px.scatter(
    pca_df_filtered,
    x='PC1',
    y='PC2',
    text='state',
    title='PCA of Infection Data in Mexico',
    labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'},
    template='plotly_white',
    height=1500
)

fig_filtered.update_traces(textposition='top center')

########

t_data = data.groupby(['date', 'sickness'], as_index=False)['value'].sum()

fig1 = px.line(t_data[t_data.sickness == 'virus papiloma humano'], x='date', y='value', color='sickness', log_y=True)

s_data = t_data[t_data.date.dt.year >= 2018].copy()

total = t_data.groupby('date', as_index=False)['value'].sum()

fig2 = px.line(total, x='date', y='value', log_y=True, title='Morbilidades totales mensuales registradas')

total_sickness = s_data.groupby('sickness', as_index=False)['value'].sum()
total_sickness.sort_values(by='value', ascending=False, inplace=True)
fig3 = px.line(s_data[s_data.sickness.isin(total_sickness[0:10].sickness)], x='date', y='value', color='sickness', log_y=True,
        title='Top 10 enfermedades de los últimos 2 años por mes')

app = Dash(__name__)

app.layout = html.Div([
    html.Div(children='Datos y Análisis IMSS'),
    html.Hr(),
    html.Div([
    dcc.Dropdown(options=list(data.sickness.unique()), value='', id='controls'),
    dcc.Graph(figure=fig1, id='controls-and-graph'),
    html.Div([
    dcc.Graph(id='controls-table', style={'display': 'inline-block', 'width':"50%"}),
    dcc.Graph(id='controls-dist', style={'display': 'inline-block', 'width':"50%"}),
    ])]),
    html.Hr(),
    dcc.Graph(figure=fig2),
    dcc.Graph(figure=fig3),
    dcc.Graph(figure=fig_filtered),
    html.Div([
        dcc.Dropdown(options=list(data.sickness.unique()), value='', id='controls_map'),
        dcc.Graph(id='map'),
    ]),
])

geo = gpd.read_file('https://raw.githubusercontent.com/angelnmara/geojson/refs/heads/master/mexicoHigh.json')
geo['name'] = geo['name'].str.lower().str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode(
    'utf-8').str.lower()
geo['state'] = geo['name']

data_p = data.replace({'distrito federal': 'ciudad de mexico',
              'sanluis potosi': 'san luis potosi'}).copy()


@callback(
    Output(component_id='map', component_property='figure'),
    Input(component_id='controls_map', component_property='value')
)
def update_map(col_chosen):
    t_data = data[data.sickness == col_chosen].copy()
    years = list(sorted(t_data.date.dt.year.unique()))
    t_data = t_data[t_data.date.dt.year.isin(years[-10:])]
    t_data = t_data.groupby([t_data.date.dt.year, 'state'])['value'].mean().to_frame()
    t_data.reset_index(inplace=True)
    t_geo = geo.merge(t_data, on="state").set_index('state')
    fig_m = px.choropleth(t_geo, geojson=t_geo.geometry, locations=t_geo.index, color='value', animation_frame='date',
                        range_color=(t_geo.value.min(), t_geo.value.max()))
    fig_m.update_geos(fitbounds="locations", visible=False)

    return fig_m


chosen = []

@callback(
    [Output(component_id='controls-and-graph', component_property='figure'),
     Output(component_id='controls-table', component_property='figure'),
     Output(component_id='controls-dist', component_property='figure')],
    Input(component_id='controls', component_property='value')
)
def update_graph(col_chosen):
    # if col_chosen == 'clear':
    #     chosen.clear()
    # else:
    #     chosen.append(col_chosen)
    chosen.clear()
    chosen.append(col_chosen)
    mask = t_data.sickness.isin(chosen)
    temp = t_data[mask]
    fig = px.line(temp, x='date', y='value', color='sickness', log_y=True,
                  title = f"Tiempo vs Morbilidades", range_x=(data.date.min(), data.date.max()),
                  labels={
                      "date": "Fecha",
                      "value": "Morbilidades Registradas"
                  }
                  )
    fig.update_layout(
        xaxis=dict(
            tickmode='linear',
            tick0= "1995-01-15",
            dtick='M12',
            tickformat='%m/%Y',
            tickangle=-45

        )
    )
    fig.update_traces(connectgaps=False)

    local_max_vals = temp.loc[temp.value == temp.value.rolling(6, center=True).max()]

    fig.add_trace(
        Scatter(
            x=local_max_vals.date,
            y=local_max_vals.value,
            name='Máximos',
            connectgaps=False,
            mode='markers+text',
            text=local_max_vals.date.dt.strftime('%b'),
            textposition='top center',
        )
    )

    figh = px.histogram(local_max_vals.date.dt.strftime('%B'), x='date', title="Distribución de Máximos",
                        labels={
                            "date": "Mes",
                        },
                        )

    figh.update_xaxes(categoryorder='total descending')
    figh.update_layout(yaxis_title="Número")

    t_temp = temp.sort_values(by='date', ascending=False)
    d = t_temp.groupby([t_temp.date.dt.strftime('%B'), t_temp.date.dt.month])['value'].mean()

    # a = d.index.to_frame(name=['month', 'month_n']).join(d).reset_index()

    a = d.to_frame().reset_index(names=['month', 'month_n']).sort_values(by='month_n', ascending=True)

    fighh = px.bar(a, x='month', y='value', title="Promedio por mes",
                         labels={
                             "month": "Mes",
                             "value": "Promedio",
                         },
                         )



    return fig, figh, fighh

server = app.server

if __name__ == "__main__":
    app.run(debug=True)
