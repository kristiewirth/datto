import base64
from io import BytesIO

import matplotlib


class FrontEnd:
    def dropdown_from_dataframe(self, name, df, chosen_col):
        """
        Create text to use for rendering an HTML dropdown from a DataFrame.

        Render by using {{ df|safe }} in your HTML file.

        Parameters
        --------
        name: str
            Name you'd like for the dropdown
        df: DataFrame
        chosen_col: str
            Which column's values will populate the dropdown

        Returns
        --------
        html_choices: str
            String you can use to render HTML

        """
        html_choices = f"""<select id="{name}" name="{name}" width: 400px;><option value="---">---</option>"""

        df.sort_values(by=chosen_col, inplace=True, ascending=True)

        for option in df[chosen_col].unique():
            html_choices += f"""<option value="{option}">{option}</option>"""
        html_choices += """</select>"""

        return html_choices

    def dataframe_to_html(self, df, title=""):
        """
        Write an entire dataframe to an HTML file with nice formatting.

        Parameters
        --------
        df: DataFrame
        title: str (optional)


        Returns
        --------
        html: str

        """

        html = """
        <html>
        <body>
            """

        min_col_widths = {col: 150 for col in df.columns}

        html += "<h2> %s </h2>\n" % title
        html += df.to_html(
            col_space=min_col_widths,
            classes="wide",
            max_rows=1000,
            escape=False,
            index=False,
        )
        html += """
        </body>
        </html>
        """

        return html

    def fig_to_html(self, fig):
        """
        Create HTML file from a matplotlib fig with workarounds for using inside a Flask app.

        Parameters
        --------
        fig: matplotlib figure

        Returns
        --------
        html: str
        """
        matplotlib.use("Agg")

        tmpfile = BytesIO()
        fig.savefig(tmpfile, format="png")
        encoded = base64.b64encode(tmpfile.getvalue()).decode("utf-8")

        html = "<img src='data:image/png;base64,{}'>".format(encoded)

        return html

