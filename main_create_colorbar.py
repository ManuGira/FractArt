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

    def __init__(self, process_name):
        self.nb_color = 2
        self.sliders_values = [0, 100]
        self.colors = ['#A911FF', '#FFF77F']
        self.current_slider = 0

        self.update_required = False
        self.color_sliders = [ComponentContext(f'color-slider-{k}-id', self.sliders_values[k]) for k in range(2)]
        self.color_picker = ComponentContext(f'color-picker-id', {'hex': self.colors[0]})

        self.app = None
        self.init_dash_app(process_name)

    def make_dash_nbColorCaption(self):
        return f"N = {self.nb_color}"

    def make_dash_slider(self, k):
        comp = self.color_sliders[k]
        return dcc.Slider(
            id=comp.id,
            min=0,
            max=1024,
            value=comp.val,
        )

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
                html.Div([
                    daq.ColorPicker(
                        id=self.color_picker.id,
                        label='Color Picker',
                        value=self.color_picker.val
                    ),
                ]),
                html.Div([
                    self.make_dash_slider(0),
                    self.make_dash_slider(1),
                ]),
            ],
        )

    def init_dash_app(self, process_name):
        self.app = dash.Dash(process_name)

        self.app.layout = html.Div([
            self.make_dash_layout(),
            html.Div(id=self.color_sliders[0].out_id, style={'display’': 'none'}),
            html.Div(id=self.color_sliders[1].out_id, style={'display’': 'none'}),
            # dcc.Interval(id='interval-id', interval=50, n_intervals=0),  # update color bar
        ])

        @self.app.callback(
            Output(component_id=self.color_picker.id, component_property='value'),
            [Input(component_id=comp.id, component_property='value') for comp in self.color_sliders])
        def callback(*values):
            print("color_sliders[0]", dash.callback_context.triggered)
            value = dash.callback_context.triggered[0]['value']
            triggered_id = dash.callback_context.triggered[0]['prop_id'][:-6]
            if value is None:
                return dash.no_update
            slider_ids = [comp.id for comp in self.color_sliders]
            print(slider_ids)
            print(triggered_id)
            self.current_slider = slider_ids.index(triggered_id)
            self.sliders_values[self.current_slider] = value
            return {'hex': self.colors[self.current_slider]}

    def start(self):
        print('Dash created')
        webbrowser.open_new('http://127.0.0.1:8050/')
        self.app.run_server(debug=True, processes=0)
        print('Dash ok')


def main(process_name):
    view_context = ViewContext(process_name)
    view_context.start()


if __name__ == '__main__':
    main(__name__)

