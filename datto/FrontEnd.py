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
        html_choices = """<select id="{}" width: 400px;><option value="---">---</option>""".format(
            name
        )
        for option in df[chosen_col].values:
            html_choices += """<option value="{0}">{0}</option>""".format(option)
        html_choices += """</select>"""

        return html_choices
