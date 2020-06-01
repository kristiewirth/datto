from datto.CleanText import CleanText

ct = CleanText()


# def test_remove_names():
#     text = "Hello John, how are you doing?"
#     cleaned_text = ct.remove_names(text)
#     assert "John" not in cleaned_text


def test_remove_links():
    text = "Here's a link: www.google.com Thanks!"
    cleaned_text = ct.remove_links(text)
    assert "www.google.com" not in cleaned_text
