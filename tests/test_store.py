from app.store import VectorStore


def test_search_returns_most_similar_first():
    store = VectorStore()
    store.add_document("doc", ["cats", "dogs", "fish"], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    hits = store.search([0.9, 0.1, 0], k=2)
    assert [h["text"] for h in hits] == ["cats", "dogs"]
    assert hits[0]["score"] > hits[1]["score"]


def test_empty_store_returns_nothing():
    assert VectorStore().search([1, 0, 0]) == []


def test_same_name_replaces_old_version():
    store = VectorStore()
    store.add_document("a.txt", ["old"], [[1, 0]])
    store.add_document("a.txt", ["new one", "new two"], [[0, 1], [1, 1]])
    assert store.documents() == [{"name": "a.txt", "chunks": 2}]


def test_delete_document():
    store = VectorStore()
    store.add_document("a.txt", ["x"], [[1, 0]])
    assert store.delete_document("a.txt") is True
    assert store.delete_document("a.txt") is False
    assert store.is_empty()


def test_data_survives_restart(tmp_path):
    path = tmp_path / "store.json"
    VectorStore(path).add_document("a.txt", ["hello"], [[1, 2]])
    reloaded = VectorStore(path)
    assert reloaded.documents() == [{"name": "a.txt", "chunks": 1}]
    assert reloaded.search([1, 2], 1)[0]["text"] == "hello"
