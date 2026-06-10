from counter import count_words, summarize


def test_counts_words():
    assert count_words("local first evals") == 3


def test_summary_shape():
    assert summarize("hello world") == {"words": 2}
