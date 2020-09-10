import webbrowser
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import numpy as np


class ResultContainer:
    """
    A simple class containing the data.
    """
    def __init__(self, p0=0, p1=0, p2=0, p3=0):
        self.p0 = p0
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3


class ModelContext:
    """
    This class contains all the measured data, and the function to process these data.
    """
    def __init__(self):
        self.x = None
        self.y = None
        self.load_data()

    def load_data(self):
        N = 1000
        p0 = (np.random.rand(1)[0]-0.5) * 8
        p1 = (np.random.rand(1)[0]-0.5) * 4
        p2 = (np.random.rand(1)[0]-0.5) * 2
        p3 = (np.random.rand(1)[0]-0.5) * 1
        self.x = (np.random.rand(N)-0.5)*5
        self.y = p0 + p1*self.x + p2*self.x**2 + p3*self.x**3 + np.random.randn(N)*0.5

    def fit_plynomial(self, degree):
        ps = np.polyfit(self.x, self.y, degree)
        return ResultContainer(*(ps[::-1]))


class ViewContext:
    PAGE_MAXWIDTH = 1000

    def __init__(self, model_context):
        self.model_context = model_context

        self.app = None
        self.init_dash_app()


    def make_dash_plot(self, fit_result=None):
        plots = []
        plots.append(
            go.Scatter(
                x=self.model_context.x,
                y=self.model_context.y,
                name='noisy measures',
                mode='markers',
                marker=dict(
                    size=3,
                    color='hsl(350,90%,50%)',
                ),
                opacity=0.7,
            ),
        )

        if fit_result is not None:
            xs = np.linspace(min(self.model_context.x), max(self.model_context.x), 200)
            ys = fit_result.p0 + fit_result.p1*xs + fit_result.p2*xs**2 + fit_result.p3*xs**3
            plots.append(
                go.Scatter(
                    x=xs,
                    y=ys,
                    name='fit',
                    line=dict(
                        width=3,
                        color='hsl(200,90%,50%)',
                    ),
                    opacity=1,
                ),
            )

        graph = dcc.Graph(
            figure={
                'data': plots,
                'layout': {
                    'title': 'polynomial fit',
                    'xaxis': {
                        'title': f'x axis'
                    },
                    'yaxis': {
                        'title': f'y axis'
                    },
                },
            },
        )
        return html.Div(className='col-6', children=[graph])

    def init_dash_app(self):
        # Create the interactive Graphs
        # https://plot.ly/dash/
        self.app = dash.Dash()
        # serve static files locally : https://dash.plot.ly/external-resources
        self.app.css.config.serve_locally = True
        self.app.scripts.config.serve_locally = True

        self.app.layout = html.Div(
            className='container',
            style={
                'max-width': f'{ViewContext.PAGE_MAXWIDTH}px',
                'margin': 'auto',
            },
            children=[
                html.Div(
                    className='header clearfix',
                    children=[
                        html.H2('Dash Example', className='text-muted'),
                        html.Hr(),
                    ],
                ),
                html.Div(
                    children=[
                        html.Div(
                            className='card-title',
                            style={'margin-top': '2em'},
                            children='Slider',
                        ),
                        dcc.Slider(
                            id='slider-id',
                            min=0,
                            max=3,
                            marks={i: str(i) for i in range(0, 4)},
                            value=1,
                        ),
                        html.Br(),
                    ]
                ),
                html.Div(
                    id='plot-id',
                    className='row',
                    children=self.make_dash_plot(),
                ),
            ],
        )

        @self.app.callback([
            Output(component_id='plot-id', component_property='children'),
        ], [
            Input(component_id='slider-id', component_property='value'),
        ])
        def phasecount_callback(slider_value):
            return self.update_view(slider_value)

    def update_view(self, slider_value):
        # Compute fit from data in model context
        fit_result = self.model_context.fit_plynomial(slider_value)

        updated_plot = self.make_dash_plot(fit_result)
        return [updated_plot]

    def start(self):
        print('Dash created')
        webbrowser.open_new('http://127.0.0.1:8050/')
        self.app.run_server(debug=False, processes=0)
        print('Dash ok')


def main():
    model_context = ModelContext()
    view_context = ViewContext(model_context)
    view_context.start()


if __name__ == '__main__':
    main()

