import base64
import io
from typing import List

import pandas as pd
from dash import Dash, dcc, html, Input, Output, State, dash_table, ctx
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go


def parse_contents(contents: str, filename: str) -> pd.DataFrame:
    _, content_string = contents.split(",", 1)
    decoded = base64.b64decode(content_string)

    if filename.lower().endswith(".csv"):
        return pd.read_csv(io.StringIO(decoded.decode("utf-8")))
    raise ValueError("Obsługiwane są tylko pliki CSV")


def get_x_series(df: pd.DataFrame, time_column: str | None):
    if time_column and time_column in df.columns:
        return df[time_column], time_column
    return pd.Series(df.index, index=df.index), "index"


app = Dash(__name__)
server = app.server

app.layout = html.Div(
    style={"fontFamily": "Arial", "padding": "16px"},
    children=[
        html.H2("GUI do etykietowania danych z sensorów"),
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "400px 1fr", "gap": "16px"},
            children=[
                html.Div(
                    style={
                        "border": "1px solid #ccc",
                        "borderRadius": "8px",
                        "padding": "12px",
                        "height": "fit-content",
                    },
                    children=[
                        dcc.Upload(
                            id="upload-data",
                            children=html.Div("Przeciągnij CSV tutaj albo kliknij, aby wybrać plik"),
                            style={
                                "width": "100%",
                                "height": "80px",
                                "lineHeight": "80px",
                                "borderWidth": "1px",
                                "borderStyle": "dashed",
                                "borderRadius": "8px",
                                "textAlign": "center",
                                "marginBottom": "12px",
                            },
                            multiple=False,
                        ),
                        html.Div(id="file-info", style={"marginBottom": "12px", "fontSize": "14px"}),
                        html.Label("Kolumna czasu (opcjonalnie)"),
                        dcc.Dropdown(id="time-column", options=[], placeholder="Wybierz kolumnę czasu albo zostaw pustą"),
                        html.Br(),
                        html.Label("Kolumny sensorów do wykresu"),
                        dcc.Dropdown(id="sensor-columns", options=[], multi=True, placeholder="Wybierz kolumny"),
                        html.Br(),
                        html.Label("Nowa etykieta"),
                        dcc.Input(id="label-input", type="text", placeholder="np. pies albo 1", style={"width": "100%"}),
                        html.Div(
                            style={"marginTop": "8px"},
                            children=[
                                dcc.RadioItems(
                                    id="label-type",
                                    options=[
                                        {"label": "string", "value": "string"},
                                        {"label": "number", "value": "number"},
                                    ],
                                    value="string",
                                    inline=True,
                                )
                            ],
                        ),
                        html.Br(),
                        html.Div(id="range-info", style={"fontSize": "14px", "marginBottom": "8px"}),
                        html.Label("Ręczna korekta indeksów"),
                        html.Div(
                            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "8px", "marginBottom": "8px"},
                            children=[
                                dcc.Input(id="manual-start-idx", type="number", placeholder="start_idx", style={"width": "100%"}),
                                dcc.Input(id="manual-end-idx", type="number", placeholder="end_idx", style={"width": "100%"}),
                            ],
                        ),
                        html.Button("Dodaj etykietę z aktualnego zakresu", id="add-label-btn", n_clicks=0, style={"marginRight": "8px"}),
                        html.Button("Usuń ostatnią etykietę", id="undo-label-btn", n_clicks=0),
                        html.Br(),
                        html.Br(),
                        html.Button("Wyczyść wszystkie etykiety", id="clear-labels-btn", n_clicks=0, style={"marginRight": "8px"}),
                        html.Label("Wartość domyślna dla nieoznaczonych próbek"),
                        dcc.Input(id="default-idx-value", type="text", value="None", style={"width": "100%", "marginBottom": "8px"}),
                        dcc.Input(id="default-label-value", type="text", value="None", style={"width": "100%", "marginBottom": "8px"}),
                        html.Label("Tryb eksportu"),
                        dcc.RadioItems(
                            id="export-mode",
                            options=[
                                {"label": "Dołącz label do oryginalnego CSV", "value": "append"},
                                {"label": "Eksportuj tylko kolumnę label", "value": "label_only"},
                            ],
                            value="append",
                            style={"marginBottom": "8px"},
                        ),
                        html.Button("Eksportuj CSV z etykietami", id="export-csv-btn", n_clicks=0, style={"marginRight": "8px"}),
                        html.Button("Eksportuj tabelę zakresów", id="export-ranges-btn", n_clicks=0),
                        dcc.Download(id="download-labeled-csv"),
                        dcc.Download(id="download-ranges-csv"),
                        html.Hr(),
                        html.Div(
                            [
                                html.Div("Jak to działa:"),
                                html.Ol(
                                    [
                                        html.Li("Wybierz kolumny do wykresu."),
                                        html.Li("Zaznacz fragment wykresu zoomem po osi X."),
                                        html.Li("Program wyliczy start_idx i end_idx."),
                                        html.Li("Popraw ręcznie indeksy, jeśli chcesz dokładniejszy zakres."),
                                        html.Li("Wpisz etykietę i kliknij dodanie etykiety."),
                                        html.Li("Przy eksporcie wszystkie indeksy w tym zakresie dostaną tę wartość w kolumnie label.")
                                    ]
                                )
                            ],
                            style={"fontSize": "14px"},
                        ),
                    ],
                ),
                html.Div(
                    children=[
                        dcc.Graph(
                            id="sensor-graph",
                            figure=go.Figure(),
                            config={"scrollZoom": True, "displaylogo": False},
                            style={"height": "650px"},
                        ),
                        html.H3("Dodane etykiety"),
                        dash_table.DataTable(
                            id="labels-table",
                            columns=[
                                {"name": "id", "id": "id"},
                                {"name": "label", "id": "label"},
                                {"name": "start_x", "id": "start"},
                                {"name": "end_x", "id": "end"},
                                {"name": "start_idx", "id": "start_idx"},
                                {"name": "end_idx", "id": "end_idx"},
                            ],
                            data=[],
                            page_size=10,
                            style_table={"overflowX": "auto"},
                            style_cell={"textAlign": "left", "padding": "8px"},
                        ),
                    ]
                ),
            ],
        ),
        dcc.Store(id="stored-data"),
        dcc.Store(id="stored-filename"),
        dcc.Store(id="stored-labels", data=[]),
        dcc.Store(id="stored-current-range"),
        dcc.Store(id="stored-current-idx-range"),
    ],
)


@app.callback(
    Output("stored-data", "data"),
    Output("stored-filename", "data"),
    Output("file-info", "children"),
    Output("time-column", "options"),
    Output("sensor-columns", "options"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    prevent_initial_call=True,
)
def handle_upload(contents, filename):
    if contents is None:
        raise PreventUpdate

    df = parse_contents(contents, filename)
    cols = [{"label": c, "value": c} for c in df.columns]
    info = f"Załadowano plik: {filename} | wiersze: {len(df)} | kolumny: {len(df.columns)}"
    return df.to_json(date_format="iso", orient="split"), filename, info, cols, cols


@app.callback(
    Output("sensor-graph", "figure"),
    Input("stored-data", "data"),
    Input("sensor-columns", "value"),
    Input("time-column", "value"),
    Input("stored-labels", "data"),
)
def update_graph(data_json, sensor_columns, time_column, labels):
    fig = go.Figure()

    if not data_json or not sensor_columns:
        fig.update_layout(title="Załaduj CSV i wybierz kolumny sensorów", dragmode="zoom", template="plotly_white")
        return fig

    df = pd.read_json(io.StringIO(data_json), orient="split")
    x, x_title = get_x_series(df, time_column)

    for col in sensor_columns:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=x, y=df[col], mode="lines", name=col))

    for item in labels or []:
        fig.add_vrect(
            x0=item["start"],
            x1=item["end"],
            annotation_text=f'{item["label"]} [{item["start_idx"]}:{item["end_idx"]}]',
            annotation_position="top left",
            opacity=0.2,
            line_width=1,
        )

    fig.update_layout(
        template="plotly_white",
        dragmode="zoom",
        xaxis_title=x_title,
        yaxis_title="value",
        legend_title="signals",
        margin={"l": 40, "r": 20, "t": 40, "b": 40},
    )
    return fig


@app.callback(
    Output("stored-current-range", "data"),
    Output("stored-current-idx-range", "data"),
    Output("range-info", "children"),
    Output("manual-start-idx", "value"),
    Output("manual-end-idx", "value"),
    Input("sensor-graph", "relayoutData"),
    State("stored-data", "data"),
    State("time-column", "value"),
    prevent_initial_call=True,
)
def capture_range(relayout_data, data_json, time_column):
    if not relayout_data or not data_json:
        raise PreventUpdate

    x0 = relayout_data.get("xaxis.range[0]")
    x1 = relayout_data.get("xaxis.range[1]")

    if x0 is None or x1 is None:
        raise PreventUpdate

    if x0 > x1:
        x0, x1 = x1, x0

    df = pd.read_json(io.StringIO(data_json), orient="split")
    x, _ = get_x_series(df, time_column)

    matched_idx = df.index[(x >= x0) & (x <= x1)].tolist()
    if not matched_idx:
        raise PreventUpdate

    start_idx = int(min(matched_idx))
    end_idx = int(max(matched_idx))
    text = f"Aktualny zakres osi X: start={x0}, end={x1} | indeksy: {start_idx}:{end_idx}"

    return {"start": x0, "end": x1}, {"start_idx": start_idx, "end_idx": end_idx}, text, start_idx, end_idx


@app.callback(
    Output("stored-labels", "data"),
    Input("add-label-btn", "n_clicks"),
    Input("undo-label-btn", "n_clicks"),
    Input("clear-labels-btn", "n_clicks"),
    State("stored-labels", "data"),
    State("stored-current-range", "data"),
    State("stored-current-idx-range", "data"),
    State("label-input", "value"),
    State("label-type", "value"),
    State("manual-start-idx", "value"),
    State("manual-end-idx", "value"),
    State("stored-data", "data"),
    prevent_initial_call=True,
)
def manage_labels(add_clicks, undo_clicks, clear_clicks, labels, current_range, current_idx_range, label_value, label_type, manual_start_idx, manual_end_idx, data_json):
    labels = labels or []
    trigger = ctx.triggered_id

    if trigger == "clear-labels-btn":
        return []

    if trigger == "undo-label-btn":
        return labels[:-1] if labels else []

    if trigger == "add-label-btn":
        if not current_range or not current_idx_range or not data_json:
            raise PreventUpdate
        if label_value is None or str(label_value).strip() == "":
            raise PreventUpdate

        if label_type == "number":
            try:
                final_value = float(label_value)
            except ValueError:
                raise PreventUpdate
        else:
            final_value = str(label_value)

        df = pd.read_json(io.StringIO(data_json), orient="split")
        n = len(df)

        start_idx = current_idx_range["start_idx"] if manual_start_idx is None else int(manual_start_idx)
        end_idx = current_idx_range["end_idx"] if manual_end_idx is None else int(manual_end_idx)

        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx

        start_idx = max(0, min(start_idx, n - 1))
        end_idx = max(0, min(end_idx, n - 1))

        new_item = {
            "id": len(labels) + 1,
            "label": final_value,
            "start": current_range["start"],
            "end": current_range["end"],
            "start_idx": start_idx,
            "end_idx": end_idx,
        }
        return labels + [new_item]

    raise PreventUpdate


@app.callback(Output("labels-table", "data"), Input("stored-labels", "data"))
def update_labels_table(labels):
    return labels or []


@app.callback(
    Output("download-labeled-csv", "data"),
    Input("export-csv-btn", "n_clicks"),
    State("stored-data", "data"),
    State("stored-filename", "data"),
    State("stored-labels", "data"),
    State("export-mode", "value"),
    State("default-label-value", "value"),
    prevent_initial_call=True,
)
def export_labeled_csv(n_clicks, data_json, filename, labels, export_mode, default_label_value):
    if not data_json:
        raise PreventUpdate

    df = pd.read_json(io.StringIO(data_json), orient="split")
    labels = labels or []

    default_value = "None" if default_label_value is None or str(default_label_value) == "" else str(default_label_value)
    label_col: List[str] = [default_value] * len(df)

    for item in labels:
        start_idx = int(item["start_idx"])
        end_idx = int(item["end_idx"])
        for idx in range(start_idx, end_idx + 1):
            if 0 <= idx < len(df):
                label_col[idx] = str(item["label"])

    safe_name = (filename or "data.csv").rsplit(".", 1)[0]

    if export_mode == "label_only":
        out = pd.DataFrame({"label": label_col})
        return dcc.send_data_frame(out.to_csv, f"{safe_name}_labels_only.csv", index=False)

    out = df.copy()
    out["label"] = label_col
    return dcc.send_data_frame(out.to_csv, f"{safe_name}_labeled.csv", index=False)


@app.callback(
    Output("download-ranges-csv", "data"),
    Input("export-ranges-btn", "n_clicks"),
    State("stored-labels", "data"),
    State("stored-filename", "data"),
    prevent_initial_call=True,
)
def export_ranges_csv(n_clicks, labels, filename):
    if not labels:
        raise PreventUpdate

    ranges_df = pd.DataFrame(labels)
    safe_name = (filename or "data.csv").rsplit(".", 1)[0]
    return dcc.send_data_frame(ranges_df.to_csv, f"{safe_name}_label_ranges.csv", index=False)


if __name__ == "__main__":
    app.run(debug=True)
