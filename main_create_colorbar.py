import webbrowser
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_daq as daq
import plotly.graph_objs as go
import numpy as np
import utils
from utils import pth
import fractal_painter
import cv2 as cv
import pickle

class ComponentContext:
    def __init__(self, id, val):
        self.id = id
        self.out_id = "out-" + id
        self.val = val

class ViewContext:
    PAGE_MAXWIDTH = 1000
    IMAGE_DIRECTORY = './assets/'
    MAX_ITER = 8196

    def __init__(self, process_name):
        self.nb_color = 10
        self.sliders_values = [170, 282, 297, 317, 231, 248, 267, 238, 327, 1024]
        self.colors = ['#000000', '#FF00BD', '#f09cf8', '#000000', '#1570ec', '#9dc2f4', '#000000', '#000000', '#20f8f0', '#000000']
        self.current_slider = 0
        self.colorbar_bgr = np.zeros((1024, 3))

        self.update_required = False
        self.color_sliders = [ComponentContext(f'color-slider-{k}-id', self.sliders_values[k]) for k in range(self.nb_color)]
        self.color_pickers = [ComponentContext(f'color-picker-{k}-id', {'hex': self.colors[k]}) for k in range(self.nb_color)]
        self.color_pickers_div = [ComponentContext(f'color-picker-div-{k}-id', '') for k in range(self.nb_color)]

        self.julia_hits = None
        self.load_julia_hits()

        self.app = None
        self.init_dash_app(process_name)

    def load_julia_hits(self):
        with open(pth(ViewContext.IMAGE_DIRECTORY, "7865.pkl"), "rb") as pickle_in:
            self.julia_hits = pickle.load(pickle_in)
        self.julia_hits = fractal_painter.fake_supersampling(self.julia_hits)
        max_iter = ViewContext.MAX_ITER
        self.julia_hits[self.julia_hits > max_iter - 1] = max_iter - 1

    def make_dash_colorpicker(self, k):
        comp_div = self.color_pickers_div[k]
        comp = self.color_pickers[k]
        return html.Div(
            id=comp_div.id,
            style=None if k == self.current_slider else {'display': 'none'},
            children=daq.ColorPicker(
                id=comp.id,
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

        colorbar = np.zeros((1, 1024, 3), dtype=np.uint8)
        for k in range(len(colors)-1):
            c0, c1 = colors[k], colors[k+1]
            p0, p1 = color_positions[k], color_positions[k+1]
            if p1-p0 == 0:
                continue
            color_section = fractal_painter.color_gradient_2(c0, c1, p1-p0)
            color_section.shape = (1,) + color_section.shape
            colorbar[0, p0:p1, :] = color_section

        self.colorbar_bgr = colorbar[0, :, ::-1]

        colorbar = cv.resize(colorbar, dsize=(1024, 100), interpolation=cv.INTER_NEAREST)

        cv.imwrite(f"{ViewContext.IMAGE_DIRECTORY}colorbar.png", colorbar[:, :, ::-1])

        plot = dcc.Graph(figure=go.Figure(go.Image(z=colorbar)))
        return plot

    def make_dash_painted_fractal(self):
        julia_bgr = fractal_painter.apply_color_map(self.julia_hits, self.colorbar_bgr)
        cv.imwrite(f"{ViewContext.IMAGE_DIRECTORY}fractal.png", julia_bgr)
        plot = dcc.Graph(figure=go.Figure(go.Image(z=julia_bgr[:, :, ::-1])))
        return plot

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
                    id='color-bar-img',
                    children=self.make_dash_colorbar()
                ),
                html.Div(
                    [self.make_dash_slider(k) for k in range(self.nb_color)],
                ),
                html.Div(
                    id='painted-fractal-img',
                    children=self.make_dash_painted_fractal()
                ),
            ],
        )

    def init_dash_app(self, process_name):
        self.app = dash.Dash(process_name)

        self.app.layout = html.Div([
            self.make_dash_layout(),
            html.Div(id=self.color_sliders[0].out_id, style={'display': 'none'}),
            html.Div(id=self.color_sliders[1].out_id, style={'display': 'none'}),
            html.Div(id=self.color_pickers[0].out_id, style={'display': 'none'}),
        ])

        @self.app.callback(
            [Output(component_id='color-bar-img', component_property='children')] +
            [Output(component_id='painted-fractal-img', component_property='children')] +
            [Output(component_id=color_picker_div.id, component_property='style') for color_picker_div in self.color_pickers_div],
            [Input(component_id=color_slider.id, component_property='value') for color_slider in self.color_sliders] +
            [Input(component_id=color_picker.id, component_property='value') for color_picker in self.color_pickers])
        def callback(*values):
            print("CALLBACK", dash.callback_context.triggered)
            value = dash.callback_context.triggered[0]['value']
            triggered_id = dash.callback_context.triggered[0]['prop_id'][:-6]
            if value is None:
                return dash.no_update

            if 'color-slider' in triggered_id:
                slider_ids = [comp.id for comp in self.color_sliders]
                self.current_slider = slider_ids.index(triggered_id)
                self.sliders_values[self.current_slider] = value
            elif 'color-picker' in triggered_id:
                self.colors[self.current_slider] = value['hex']

            print(f"self.sliders_values = {self.sliders_values}")
            print(f"self.colors = {self.colors}")

            colorbar_styles = []
            for k in range(self.nb_color):
                if self.current_slider == k:
                    colorbar_styles.append(None)
                else:
                    colorbar_styles.append({'display': 'none'})

            return [self.make_dash_colorbar(), self.make_dash_painted_fractal()] + colorbar_styles

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

