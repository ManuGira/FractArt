import webbrowser
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_daq as daq
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go
import numpy as np


class ComponentContext:
    def __init__(self, id, val):
        self.id = id
        self.out_id = "out-" + id
        self.val = val

class ViewContext:
    PAGE_MAXWIDTH = 1000

    def __init__(self):
        self.nb_color = 2
        self.sliders_values = [0, 100]

        self.update_required = False
        self.btn_minus = ComponentContext('btn-minus-id', 0)
        self.btn_plus = ComponentContext('btn-plus-id', 0)
        self.color_sliders = []

        self.app = None
        self.init_dash_app()

    def make_dash_nbColorCaption(self):
        return f"N = {self.nb_color}"

    def make_dash_sliders(self):
        for k in range(self.nb_color):
            val = self.sliders_values[k]
            comp = ComponentContext('btn-plus-id', 0)
            id = "colorbar-sliders-id",
            children = [
                           dcc.Slider(
                               id="colorbar-slider-id",
                               min=0,
                               max=1023,
                               value=255,
                           ),
                       ],

    def make_dash_layout(self):
        return html.Div(
            id='layout-id',
            className='container',
            style={
                'max-width': f'{ViewContext.PAGE_MAXWIDTH}px',
                'margin': 'auto',
            },
            children=[
                html.Div(
                    className='header clearfix',
                    children=[
                        html.H2('Colorbar creator', className='text-muted'),
                        html.Hr(),
                    ],
                ),
                html.Div(
                    children=[
                        html.Div(id="nb-color-id", children=self.make_dash_nbColorCaption()),
                        html.Button(children='-', id=self.btn_minus.id, n_clicks=self.btn_minus.val),
                        html.Button(children='+', id=self.btn_plus.id, n_clicks=self.btn_plus.val),
                    ],
                ),
                html.Div([
                    daq.ColorPicker(
                        id='my-color-picker',
                        label='Color Picker',
                        value=dict(hex='#119DFF')
                    ),
                ]),
                html.Div(
                    self.make_dash_sliders(),
                ),
            ],
        )

    def init_dash_app(self):
        # Create the interactive Graphs
        # https://plot.ly/dash/
        self.app = dash.Dash()
        # serve static files locally : https://dash.plot.ly/external-resources
        # self.app.css.config.serve_locally = True
        # self.app.scripts.config.serve_locally = True

        self.app.layout = html.Div([
            self.make_dash_layout(),
            html.Div(id=self.btn_minus.out_id, style={'display’': 'none'}),
            html.Div(id=self.btn_plus.out_id, style={'display’': 'none'}),
            html.Div(id='out-colorbar-slider-id', style={'display’': 'none'}),
            dcc.Interval(id='interval-id', interval=50, n_intervals=0),
        ])

        @self.app.callback(
                Output(component_id=self.btn_minus.out_id, component_property='children'),
                Input(component_id=self.btn_minus.id, component_property='n_clicks'),)
        def btn_minus_callback(val):
            print("btn_minus_callback", dash.callback_context.triggered)
            if val == self.btn_minus.val:
                return dash.no_update
            self.nb_color = max(0, self.nb_color - 1)
            self.btn_minus.val = val
            self.update_required = True
            return dash.no_update

        @self.app.callback(
            Output(component_id=self.btn_plus.out_id, component_property='children'),
            Input(component_id=self.btn_plus.id, component_property='n_clicks'), )
        def btn_minus_callback(val):
            print("btn_plus_callback", dash.callback_context.triggered)
            if val == self.btn_plus.val:
                return dash.no_update
            self.nb_color += 1
            self.btn_plus.val = val
            self.update_required = True
            return dash.no_update

        # @self.app.callback(
        #     Output(component_id='out-colorbar-slider-id', component_property='children'),
        #     Input(component_id='colorbar-slider-id', component_property='value'), )
        # def colorbar_slider_callback(value):
        #     print("colorbar_slider_callback", dash.callback_context.triggered)
        #     print(dash.callback_context)
        #     return dash.no_update

        @self.app.callback(
            Output(component_id='layout-id', component_property='children'),
            Input(component_id='interval-id', component_property='n_intervals'), )
        def interval_callback(n_intervals):
            # print("interval_callback", dash.callback_context.triggered)
            if self.update_required:
                self.update_required = False
                return self.make_dash_layout()
            else:
                return dash.no_update

    def start(self):
        print('Dash created')
        webbrowser.open_new('http://127.0.0.1:8050/')
        self.app.run_server(debug=True, processes=0)
        print('Dash ok')


def main():
    view_context = ViewContext()
    view_context.start()


if __name__ == '__main__':
    main()

