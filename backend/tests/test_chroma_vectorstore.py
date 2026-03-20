import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.core.vectorstore.chroma import ChromaVectorStore
from app.models.chunk import Chunk


@pytest.fixture
def mock_embedding_provider():
    """EmbeddingProvider のモック"""
    provider = AsyncMock()
    provider.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])
    provider.embed_batch = AsyncMock(
        return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    )
    return provider


@pytest.fixture
def sample_chunks():
    """テスト用サンプルチャンク"""
    return [
        Chunk(
            content="First chunk content",
            document_id="doc1",
            source_file="test.md",
            heading_path=["Title"],
            page_number=None,
            chunk_index=0,
            token_count=4,
        ),
        Chunk(
            content="Second chunk content",
            document_id="doc1",
            source_file="test.md",
            heading_path=["Title", "Section"],
            page_number=None,
            chunk_index=1,
            token_count=4,
        ),
    ]


@pytest.fixture
def vectorstore(mock_embedding_provider, tmp_path):
    """ChromaVectorStore インスタンス"""
    return ChromaVectorStore(
        embedding_provider=mock_embedding_provider,
        persist_directory=str(tmp_path),
    )


@pytest.mark.asyncio
async def test_add_chunks(vectorstore, sample_chunks):
    """チャンク追加が正しく動作する"""
    with patch.object(vectorstore.client, "get_or_create_collection") as mock_collection:
        mock_coll = MagicMock()
        mock_collection.return_value = mock_coll

        await vectorstore.add_chunks(sample_chunks, "test_collection")

        # コレクション取得確認
        mock_collection.assert_called_once_with(name="test_collection")

        # collection.add() が正しい引数で呼ばれたか
        mock_coll.add.assert_called_once()
        call_args = mock_coll.add.call_args
        assert call_args[1]["ids"] == ["doc1:0", "doc1:1"]
        assert len(call_args[1]["embeddings"]) == 2
        assert call_args[1]["documents"] == [
            "First chunk content",
            "Second chunk content",
        ]
        assert call_args[1]["metadatas"][0]["source_file"] == "test.md"
        assert call_args[1]["metadatas"][0]["heading_path"] == "Title"
        assert call_args[1]["metadatas"][1]["heading_path"] == "Title|Section"


@pytest.mark.asyncio
async def test_add_chunks_empty_list(vectorstore):
    """空のチャンクリストは無視される"""
    with patch.object(vectorstore.client, "get_or_create_collection") as mock_collection:
        await vectorstore.add_chunks([], "test_collection")

        # get_or_create_collection は呼ばれない
        mock_collection.assert_not_called()


@pytest.mark.asyncio
async def test_search(vectorstore, sample_chunks):
    """クエリ検索が正しく動作する"""
    with patch.object(vectorstore.client, "get_or_create_collection") as mock_collection:
        mock_coll = MagicMock()
        mock_collection.return_value = mock_coll

        # モック検索結果
        mock_coll.query.return_value = {
            "ids": [["doc1:0", "doc1:1"]],
            "documents": [["First chunk content", "Second chunk content"]],
            "distances": [[0.1, 0.2]],
            "metadatas": [
                [
                    {
                        "source_file": "test.md",
                        "document_id": "doc1",
                        "chunk_index": 0,
                        "page_number": -1,
                        "token_count": 4,
                        "heading_path": "Title",
                    },
                    {
                        "source_file": "test.md",
                        "document_id": "doc1",
                        "chunk_index": 1,
                        "page_number": -1,
                        "token_count": 4,
                        "heading_path": "Title|Section",
                    },
                ]
            ],
        }

        results = await vectorstore.search("test query", "test_collection", limit=5)

        # 検索結果確認
        assert len(results) == 2
        assert results[0].content == "First chunk content"
        assert results[0].heading_path == ["Title"]
        assert results[1].content == "Second chunk content"
        assert results[1].heading_path == ["Title", "Section"]
        assert results[0].page_number is None


@pytest.mark.asyncio
async def test_search_with_page_numbers(vectorstore):
    """PDF のページ番号が正しく復元される"""
    with patch.object(vectorstore.client, "get_or_create_collection") as mock_collection:
        mock_coll = MagicMock()
        mock_collection.return_value = mock_coll

        mock_coll.query.return_value = {
            "ids": [["doc2:0"]],
            "documents": [["PDF page 1"]],
            "distances": [[0.0]],
            "metadatas": [
                [
                    {
                        "source_file": "report.pdf",
                        "document_id": "doc2",
                        "chunk_index": 0,
                        "page_number": 1,
                        "token_count": 2,
                        "heading_path": "",
                    }
                ]
            ],
        }

        results = await vectorstore.search("query", "test_collection")

        assert len(results) == 1
        assert results[0].page_number == 1
        assert results[0].heading_path == []


@pytest.mark.asyncio
async def test_search_no_results(vectorstore):
    """検索結果なしの場合、空のリストを返す"""
    with patch.object(vectorstore.client, "get_or_create_collection") as mock_collection:
        mock_coll = MagicMock()
        mock_collection.return_value = mock_coll

        mock_coll.query.return_value = {
            "ids": [[]],
            "documents": [[]],
            "distances": [[]],
            "metadatas": [[]],
        }

        results = await vectorstore.search("query", "test_collection")

        assert results == []


@pytest.mark.asyncio
async def test_delete_chunk(vectorstore):
    """チャンク削除が正しく動作する"""
    with patch.object(vectorstore.client, "get_or_create_collection") as mock_collection:
        mock_coll = MagicMock()
        mock_collection.return_value = mock_coll

        await vectorstore.delete_chunk("doc1:0", "test_collection")

        mock_coll.delete.assert_called_once_with(ids=["doc1:0"])


@pytest.mark.asyncio
async def test_delete_collection(vectorstore):
    """コレクション削除が正しく動作する"""
    with patch.object(vectorstore.client, "delete_collection") as mock_delete:
        await vectorstore.delete_collection("test_collection")

        mock_delete.assert_called_once_with(name="test_collection")


@pytest.mark.asyncio
async def test_update_chunk(vectorstore):
    """チャンク更新は削除と追加の組み合わせ"""
    chunk = Chunk(
        content="Updated content",
        document_id="doc1",
        source_file="test.md",
        heading_path=[],
        page_number=None,
        chunk_index=0,
        token_count=2,
    )

    with patch.object(vectorstore, "delete_chunk", new_callable=AsyncMock) as mock_delete, \
         patch.object(vectorstore, "add_chunks", new_callable=AsyncMock) as mock_add:

        await vectorstore.update_chunk(chunk, "test_collection")

        mock_delete.assert_called_once_with("doc1:0", "test_collection")
        mock_add.assert_called_once_with([chunk], "test_collection")
