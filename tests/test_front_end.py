from hypothesis import given
from hypothesis.extra.pandas import column, data_frames

from datto.FrontEnd import FrontEnd

fe = FrontEnd()


@given(
    data_frames(
        columns=[column(name="text", dtype=str), column(dtype=int), column(dtype=bool),]
    )
)
def test_dropdown_from_dataframe(df):
    html_text = fe.dropdown_from_dataframe("Choose an option", df, "text")

    assert "<select" in html_text
