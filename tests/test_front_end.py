import matplotlib.pyplot as plt
import pandas as pd
from datto.FrontEnd import FrontEnd
from hypothesis import given
from hypothesis.extra.pandas import column, data_frames

fe = FrontEnd()


@given(
    data_frames(
        columns=[column(name="text", dtype=str), column(dtype=int), column(dtype=bool),]
    )
)
def test_dropdown_from_dataframe(df):
    html_text = fe.dropdown_from_dataframe("Choose an option", df, "text")

    assert "<select" in html_text


@given(
    data_frames(
        columns=[column(name="text", dtype=str), column(dtype=int), column(dtype=bool),]
    )
)
def test_dropdown_from_dataframe(df):
    df_html = fe.dataframe_to_html(df)

    assert "<html>" in df_html


def test_fig_to_html():
    df = pd.DataFrame(["cat1", "cat2", "cat2", "cat3"], columns=["categories"])

    fig = plt.figure()
    fig = df["categories"].hist().get_figure()

    fig_html = fe.fig_to_html(fig)

    assert "img src" in fig_html
