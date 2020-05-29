from datto.CleanText import CleanText

testCleanText = CleanText()


def test_removeNames():
    text = "Hello John, how are you doing?"
    cleaned_text = testCleanText.removePiiNames(text)
    assert "John" not in cleaned_text


def test_removeLinks():
    text = "Here's a link: www.google.com Thanks!"
    cleaned_text = testCleanText.removeLinks(text)
    assert "www.google.com" not in cleaned_text
