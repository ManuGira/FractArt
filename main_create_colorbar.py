import webbrowser
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_daq as daq
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go
import numpy as np
import utils
import fractal_painter
import cv2 as cv
import flask

class ComponentContext:
    def __init__(self, id, val):
        self.id = id
        self.out_id = "out-" + id
        self.val = val

class ViewContext:
    PAGE_MAXWIDTH = 1000
    STATIC_IMAGE_ROUTE = '/static/'
    IMAGE_DIRECTORY = './assets/'

    def __init__(self, process_name):
        self.nb_color = 3
        self.sliders_values = [0, 100, 120]
        self.colors = ['#A911FF', '#FFF77F', '#22F722']
        self.current_slider = 0

        self.update_required = False
        self.color_sliders = [ComponentContext(f'color-slider-{k}-id', self.sliders_values[k]) for k in range(self.nb_color)]
        self.color_pickers = [ComponentContext(f'color-picker-{k}-id', {'hex': self.colors[k]}) for k in range(self.nb_color)]

        self.app = None
        self.init_dash_app(process_name)

    def make_dash_nbColorCaption(self):
        return f"N = {self.nb_color}"

    def make_dash_colorpicker(self, k):
        comp = self.color_pickers[k]
        return html.Div(
            id=comp.id,
            style=None if k == self.current_slider else {'display': 'none'},
            children=daq.ColorPicker(
                label=f'Color Picker {k}',
                value=comp.val,
            ),
        )

    def make_dash_slider(self, k):
        comp = self.color_sliders[k]
        return dcc.Slider(
            id=comp.id,
            min=0,
            max=1024,
            value=comp.val,
        )

    def make_dash_colorbar(self):
        order = np.argsort(self.sliders_values)
        color_positions = [self.sliders_values[o] for o in order]
        colors = [self.colors[o] for o in order]

        color_positions = [0] + color_positions + [1023]
        colors = colors[:1] + colors + colors[-1:]
        colors = [utils.color_hex2rgb(c) for c in colors]
        # colors = [c[::-1] for c in colors]  # rgb to bgr

        colorbar = np.zeros((1, 1024, 3), dtype=np.uint8)
        for k in range(len(colors)-1):
            c0, c1 = colors[k], colors[k+1]
            p0, p1 = color_positions[k], color_positions[k+1]
            if p1-p0 == 0:
                continue
            color_section = fractal_painter.color_gradient([c0, c1], p1-p0)
            color_section.shape = (1,) + color_section.shape
            colorbar[0, p0:p1, :] = color_section
        colorbar = cv.resize(colorbar, dsize=(1024, 100), interpolation=cv.INTER_NEAREST)
        cv.imwrite(f"{ViewContext.IMAGE_DIRECTORY}colorbar.png", colorbar)

        return dcc.Graph(figure=go.Figure(go.Image(z=colorbar)), style={"width": "100%"})  #, "display": "inline-block"})

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
                    [self.make_dash_colorpicker(k) for k in range(self.nb_color)],
                ),
                html.Div(
                    [self.make_dash_slider(k) for k in range(self.nb_color)],
                ),
                html.Div(
                    id='color-bar-img',
                    children=self.make_dash_colorbar()
                    # html.Img(id='colorbar-img-id'),
                ),
            ],
        )

    def init_dash_app(self, process_name):
        self.make_dash_colorbar()

        self.app = dash.Dash(process_name)

        self.app.layout = html.Div([
            self.make_dash_layout(),
            html.Div(id=self.color_sliders[0].out_id, style={'display': 'none'}),
            html.Div(id=self.color_sliders[1].out_id, style={'display': 'none'}),
            html.Div(id=self.color_pickers[0].out_id, style={'display': 'none'}),
            # dcc.Interval(id='interval-id', interval=50, n_intervals=0),  # update color bar
        ])

        @self.app.callback(
            [Output(component_id='color-bar-img', component_property='children')],
            [Output(component_id=color_picker.id, component_property='style') for color_picker in self.color_pickers] +
            [Input(component_id=color_slider.id, component_property='value') for color_slider in self.color_sliders])
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

            colorbar_styles = []
            for k in range(self.nb_color):
                if self.current_slider == k:
                    colorbar_styles.append(None)
                else:
                    colorbar_styles.append({'display': 'none'})

            return [self.make_dash_colorbar()] + colorbar_styles


    def start(self):
        print('Dash created')
        webbrowser.open_new('http://127.0.0.1:8050/')
        self.app.run_server(debug=False, processes=0)
        print('Dash ok')


def main(process_name):
    view_context = ViewContext(process_name)
    view_context.start()


if __name__ == '__main__':
    main(__name__)

