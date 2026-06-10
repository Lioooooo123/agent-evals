import string_tools


def test_adds_numbers():
    assert string_tools.add(2, 3) == 5


def test_normalizes_title_case():
    assert string_tools.normalize_title("  hello world  ") == "Hello World"


def test_slugify_text():
    assert string_tools.slugify("Hello, Local First!") == "hello-local-first"
