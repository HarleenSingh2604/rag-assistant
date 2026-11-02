import pytest

from app.chunking import split_text


def test_empty_text_gives_no_chunks():
    assert split_text("   \n\n  ") == []


def test_short_text_is_one_chunk():
    assert split_text("Hello world.", 800, 100) == ["Hello world."]


def test_paragraphs_are_packed_up_to_the_limit():
    text = "A" * 300 + "\n\n" + "B" * 300 + "\n\n" + "C" * 300
    chunks = split_text(text, chunk_size=700, overlap=50)
    assert len(chunks) == 2
    assert all(len(c) <= 700 for c in chunks)


def test_long_paragraph_is_split_with_overlap():
    words = " ".join(f"word{i}" for i in range(400))
    chunks = split_text(words, chunk_size=200, overlap=40)
    assert len(chunks) > 2
    assert all(len(c) <= 200 for c in chunks)
    # neighbouring chunks share some text
    assert any(w in chunks[1] for w in chunks[0].split()[-3:])
    # nothing is lost
    assert "word0" in chunks[0] and "word399" in chunks[-1]


def test_invalid_overlap_is_rejected():
    with pytest.raises(ValueError):
        split_text("text", chunk_size=100, overlap=100)
