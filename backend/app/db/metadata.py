import aiosqlite
from datetime import datetime
from dataclasses import dataclass


@dataclass
class Document:
    """ドキュメントメタデータ"""

    id: str
    collection_name: str
    filename: str
    file_size: int
    chunk_count: int
    created_at: str
    updated_at: str


@dataclass
class Collection:
    """コレクションメタデータ"""

    name: str
    description: str
    document_count: int
    created_at: str
    updated_at: str


class MetadataDB:
    """SQLiteメタデータ管理"""

    def __init__(self, db_path: str = "./metadata.db"):
        self.db_path = db_path

    async def init(self):
        """テーブル初期化"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS collections (
                    name TEXT PRIMARY KEY,
                    description TEXT,
                    document_count INTEGER DEFAULT 0,
                    created_at TEXT,
                    updated_at TEXT
                )
                """
            )
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    collection_name TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    file_size INTEGER,
                    chunk_count INTEGER,
                    created_at TEXT,
                    updated_at TEXT,
                    FOREIGN KEY (collection_name) REFERENCES collections(name)
                )
                """
            )
            await db.commit()

    async def create_collection(
        self, name: str, description: str = ""
    ) -> Collection:
        """コレクション作成"""
        now = datetime.utcnow().isoformat()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT OR IGNORE INTO collections (name, description, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (name, description, now, now),
            )
            await db.commit()

        return Collection(
            name=name,
            description=description,
            document_count=0,
            created_at=now,
            updated_at=now,
        )

    async def add_document(
        self, document_id: str, collection_name: str, filename: str, file_size: int, chunk_count: int
    ) -> Document:
        """ドキュメント追加"""
        now = datetime.utcnow().isoformat()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO documents (id, collection_name, filename, file_size, chunk_count, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (document_id, collection_name, filename, file_size, chunk_count, now, now),
            )
            # コレクションのドキュメント数を更新
            await db.execute(
                "UPDATE collections SET document_count = document_count + 1 WHERE name = ?",
                (collection_name,),
            )
            await db.commit()

        return Document(
            id=document_id,
            collection_name=collection_name,
            filename=filename,
            file_size=file_size,
            chunk_count=chunk_count,
            created_at=now,
            updated_at=now,
        )

    async def get_documents(self, collection_name: str) -> list[Document]:
        """コレクション内の全ドキュメント取得"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT id, collection_name, filename, file_size, chunk_count, created_at, updated_at FROM documents WHERE collection_name = ?",
                (collection_name,),
            )
            rows = await cursor.fetchall()

        return [
            Document(
                id=row[0],
                collection_name=row[1],
                filename=row[2],
                file_size=row[3],
                chunk_count=row[4],
                created_at=row[5],
                updated_at=row[6],
            )
            for row in rows
        ]

    async def get_collections(self) -> list[Collection]:
        """全コレクション取得"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT name, description, document_count, created_at, updated_at FROM collections"
            )
            rows = await cursor.fetchall()

        return [
            Collection(
                name=row[0],
                description=row[1],
                document_count=row[2],
                created_at=row[3],
                updated_at=row[4],
            )
            for row in rows
        ]

    async def delete_document(self, document_id: str):
        """ドキュメント削除"""
        async with aiosqlite.connect(self.db_path) as db:
            # 削除前にコレクション名を取得
            cursor = await db.execute(
                "SELECT collection_name FROM documents WHERE id = ?",
                (document_id,),
            )
            row = await cursor.fetchone()
            if row:
                collection_name = row[0]
                await db.execute(
                    "DELETE FROM documents WHERE id = ?",
                    (document_id,),
                )
                # コレクションのドキュメント数を更新
                await db.execute(
                    "UPDATE collections SET document_count = document_count - 1 WHERE name = ?",
                    (collection_name,),
                )
            await db.commit()

    async def delete_collection(self, collection_name: str):
        """コレクション削除（関連ドキュメントも削除）"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "DELETE FROM documents WHERE collection_name = ?",
                (collection_name,),
            )
            await db.execute(
                "DELETE FROM collections WHERE name = ?",
                (collection_name,),
            )
            await db.commit()
