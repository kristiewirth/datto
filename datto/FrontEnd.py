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
