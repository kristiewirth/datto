import pandas as pd
from datto.FrontEnd import FrontEnd

fe = FrontEnd()

df = pd.DataFrame(["opt1", "opt2", "opt3"], columns=["text"])


def test_dropdown_from_dataframe():
    html_text = fe.dropdown_from_dataframe("Choose an option", df, "text")

    assert "<select" in html_text

