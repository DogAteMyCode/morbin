import locale
# locale.setlocale(locale.LC_TIME, 'es_MX')
from dash import Dash, html, dcc, callback, Output, Input
from sklearn.preprocessing import StandardScaler
from plotly.graph_objs import Scatter
from sklearn.decomposition import PCA
import plotly.express as px
import geopandas as gpd
import pandas as pd



#### Data For Resources

file_path = "Unified_Cleaned_Data.zip"
data_resource = pd.read_csv(file_path, encoding='latin1', low_memory=False)

data_resource['Nombre Estado'] = data_resource['Nombre Estado'].str.strip()
data_resource = data_resource[~data_resource['Nombre Estado'].isin(['Desconocido'])]
data_resource['Nombre Estado'] = data_resource['Nombre Estado'].replace({'MÉXICO': 'Estado de México'})

camas_data_by_year = data_resource.groupby(['Año', 'Nombre Estado'])['Total camas area hospitalización'].sum().reset_index()

fig_camas_by_year = px.bar(
    camas_data_by_year,
    x='Nombre Estado',
    y='Total camas area hospitalización',
    color='Nombre Estado',
    animation_frame='Año',  # Añadir el filtro por año
    title='Distribución de Camas por Estado y Año',
    labels={'Total camas area hospitalización': 'Total de Camas', 'Nombre Estado': 'Estado'},
    color_discrete_sequence=px.colors.qualitative.Set2,
    template="plotly_dark"
)
options = dict(
    xaxis={'categoryorder': 'total descending'},
    updatemenus=[dict(type='buttons',
                      showactive=False,
                      y=-1.75,
                      x=0.01,
                      xanchor='left',
                      yanchor='bottom')
                 ]
)
fig_camas_by_year.update_layout(
    **options,
    showlegend=False,
)

fig_camas_by_year.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 2000
fig_camas_by_year['layout']['sliders'][0]['pad']=dict(r= 10, t= 150.0,)

consultorios_columns = [col for col in data_resource.columns if 'Consultorios' in col]
consultorios_data_by_year = data_resource[consultorios_columns + ['Nombre Estado', 'Año']].groupby(
    ['Año', 'Nombre Estado']).sum().reset_index()

consultorios_melted = consultorios_data_by_year.melt(
    id_vars=['Año', 'Nombre Estado'],
    var_name='Especialidad',
    value_name='Cantidad'
)

fig_consultorios_by_year = px.bar(
    consultorios_melted,
    x='Nombre Estado',
    y='Cantidad',
    color='Especialidad',
    animation_frame='Año',
    title='Distribución de Consultorios por Especialidad y Estado a lo Largo de los Años',
    labels={'Nombre Estado': 'Estado', 'Cantidad': 'Número de Consultorios'},
    range_y= (0, 5000)
)

fig_consultorios_by_year.update_layout(
    **options
)

fig_consultorios_by_year.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 2000
fig_consultorios_by_year['layout']['sliders'][0]['pad']=dict(r= 10, t= 150.0,)

consultas_data = data_resource.groupby('Año')['Total de consultorios'].sum().reset_index()
fig_consultas = px.line(
    consultas_data,
    x='Año',
    y='Total de consultorios',
    title='Tendencia de Consultorios Disponibles por Año',
    labels={'Año': 'Año', 'Total de consultorios': 'Número de Consultorios'},
    markers=True
)

equipamiento_columns = [
    'Tomógrafos computados', 'Ultrasonido', 'Mastógrafos (analógico y digital)', 'Equipos de resonancia magnética'
]

for col in equipamiento_columns:
    if col in data_resource.columns:
        data_resource[col] = pd.to_numeric(data_resource[col], errors='coerce').fillna(0)


equipamiento_data_by_year = data_resource.groupby(['Año', 'Nombre Estado'])[equipamiento_columns].sum().reset_index()

equipamiento_melted = equipamiento_data_by_year.melt(
    id_vars=['Año', 'Nombre Estado'],
    var_name='Equipamiento',
    value_name='Cantidad'
)

fig_equipamiento_by_year = px.bar(
    equipamiento_melted,
    x='Nombre Estado',
    y='Cantidad',
    color='Equipamiento',
    animation_frame='Año',
    title='Distribución de Equipamiento Médico por Estado y Año',
    labels={'Nombre Estado': 'Estado', 'Cantidad': 'Cantidad de Equipos'},
    range_y=(0, 1000)
)

fig_equipamiento_by_year.update_layout(
    **options
)
fig_equipamiento_by_year.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 2000
fig_equipamiento_by_year['layout']['sliders'][0]['pad']=dict(r= 10, t= 150.0,)

state_counts_by_year = data_resource.groupby(['Año', 'Nombre Estado']).size().reset_index(name='Número de Casos')

state_counts_by_year = (
    state_counts_by_year.sort_values(by=['Año', 'Número de Casos'], ascending=[True, False])
    .groupby('Año')
    .head(10)
)

fig_state_counts_by_year = px.bar(
    state_counts_by_year,
    x='Nombre Estado',
    y='Número de Casos',
    color='Nombre Estado',
    animation_frame='Año',
    title='Top 10 Estados con Mayor Uso de Servicios por Año',
    labels={'Nombre Estado': 'Estado', 'Número de Casos': 'Número de Casos'},
    color_discrete_sequence=px.colors.qualitative.Pastel
)

# Ajustar diseño del gráfico
fig_state_counts_by_year.update_layout(
    **options
)

#### Data For Sickness

data = pd.read_csv('new.zip', parse_dates=['date'])

t_data = data.groupby(['date', 'sickness'], as_index=False)['value'].sum()

fig1 = px.line(t_data[t_data.sickness == 'virus papiloma humano'], x='date', y='value', color='sickness', log_y=True)

s_data = t_data[t_data.date.dt.year >= 2018].copy()

total = t_data.groupby('date', as_index=False)['value'].sum()

fig2 = px.line(total, x='date', y='value', log_y=True, title='Morbilidades totales mensuales registradas')

total_sickness = s_data.groupby('sickness', as_index=False)['value'].sum()
total_sickness.sort_values(by='value', ascending=False, inplace=True)
fig3 = px.line(s_data[s_data.sickness.isin(total_sickness[0:10].sickness)], x='date', y='value', color='sickness',
               log_y=True,
               title='Top 10 enfermedades de los últimos 2 años por mes')

app = Dash(__name__)

app.layout = html.Div([
    html.Nav([
        dcc.Dropdown(options=[{'label': "Morbiliad", "value": 1}, {"label": "Recursos", "value": 2}], id="scene",
                     value=1, className="nav-link"),
    ]),
    html.Hr(),
    html.Div([
        html.Div(children='Datos y Análisis IMSS'),
        html.Hr(),
        html.Div([
            dcc.Dropdown(options=list(data.sickness.unique()), value='', id='controls', className="form-select"),
            dcc.Graph(id='controls-and-graph'),
            html.Div([
                dcc.Graph(id='controls-table', style={'display': 'inline-block', 'width': "50%"}),
                dcc.Graph(id='controls-dist', style={'display': 'inline-block', 'width': "50%"}),
            ])]),
        html.Hr(),
        dcc.Graph(),
        dcc.Graph(),
        html.Div([
            dcc.Dropdown(options=list(data.sickness.unique()), value='', id='controls_map', className="form-select"),
            dcc.Graph(id='map'),
        ]),
    ], id='sickness_layout'),
    html.Div(
        children=[
            dcc.Graph(),
            dcc.Graph(),
            dcc.Graph(),
            dcc.Graph(),
            dcc.Graph()
        ],
        id='resource_layout'
    )
])

geo = gpd.read_file('https://raw.githubusercontent.com/angelnmara/geojson/refs/heads/master/mexicoHigh.json')
geo['name'] = geo['name'].str.lower().str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode(
    'utf-8').str.lower()
geo['state'] = geo['name']

data_p = data.replace({'distrito federal': 'ciudad de mexico',
                       'sanluis potosi': 'san luis potosi'}).copy()


def sickness_scene():
    return [
        html.Div(children='Datos y Análisis IMSS', style={'padding-left': "50px"}),
        html.Hr(),
        html.Div([
            dcc.Dropdown(options=list(data.sickness.unique()), value='', id='controls', className="form-select"),
            dcc.Graph(figure=fig1, id='controls-and-graph'),
            html.Div([
                dcc.Graph(id='controls-table', style={'display': 'inline-block', 'width': "50%"}),
                dcc.Graph(id='controls-dist', style={'display': 'inline-block', 'width': "50%"}),
            ])]),
        html.Hr(),
        dcc.Graph(figure=fig2),
        dcc.Graph(figure=fig3),
        html.Div([
            dcc.Dropdown(options=list(data.sickness.unique()), value='', id='controls_map', className="form-select"),
            dcc.Graph(id='map'),
        ]),
    ], [

    ]


def resource_scene():
    return [

    ], [
        html.Div(children='Datos y Análisis de Recursos', style={'padding-left': "50px"}),
        html.Hr(),
        dcc.Graph(figure=fig_camas_by_year),
        dcc.Graph(figure=fig_consultorios_by_year),
        dcc.Graph(figure=fig_consultas),
        dcc.Graph(figure=fig_equipamiento_by_year),
        dcc.Graph(figure=fig_state_counts_by_year)
    ]


@callback(
    [Output(component_id='sickness_layout', component_property='children'),
     Output(component_id='resource_layout', component_property='children')],
    [Input(component_id='scene', component_property='value')],
)
def scene_changer(value):
    if value == 1:
        return sickness_scene()
    elif value == 2:
        return resource_scene()
    else:
        return sickness_scene()


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
                  title=f"Tiempo vs Morbilidades", range_x=(data.date.min(), data.date.max()),
                  labels={
                      "date": "Fecha",
                      "value": "Morbilidades Registradas"
                  }
                  )
    fig.update_layout(
        xaxis=dict(
            tickmode='linear',
            tick0="1995-01-15",
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
