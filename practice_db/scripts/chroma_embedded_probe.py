"""Индексация во встроенной ChromaDB (PersistentClient) в отдельном процессе.

Отдельный процесс нужен для корректного измерения памяти: в процессе ноутбука уже загружены
модель эмбеддингов и другие библиотеки, и прирост RSS не отражал бы потребление самой БД.

Аргументы: каталог хранилища, файл векторов .npy, файл метаданных .json, число документов,
имя коллекции, размер пакета, параметры HNSW. Результат выводится в stdout в формате JSON.
"""
import argparse
import json
import shutil
import threading
import time
from pathlib import Path

import numpy as np
import psutil

parser = argparse.ArgumentParser()
parser.add_argument("--path", required=True)
parser.add_argument("--vectors", required=True)
parser.add_argument("--payloads", required=True)
parser.add_argument("--n", type=int, required=True)
parser.add_argument("--collection", required=True)
parser.add_argument("--batch", type=int, default=500)
parser.add_argument("--m", type=int, default=16)
parser.add_argument("--ef-construct", type=int, default=100)
parser.add_argument("--ef-search", type=int, default=100)
args = parser.parse_args()

proc = psutil.Process()
vectors = np.load(args.vectors)[: args.n]
payloads = json.loads(Path(args.payloads).read_text(encoding="utf-8"))[: args.n]

import chromadb  # noqa: E402  (импорт после загрузки данных: базовая линия включает библиотеку)

path = Path(args.path)
if path.exists():
    shutil.rmtree(path)
rss_before = proc.memory_info().rss / 2**20

peak = {"rss": rss_before}
stop = threading.Event()


def sample():
    while not stop.is_set():
        peak["rss"] = max(peak["rss"], proc.memory_info().rss / 2**20)
        time.sleep(0.05)


sampler = threading.Thread(target=sample, daemon=True)
sampler.start()
t0 = time.perf_counter()
client = chromadb.PersistentClient(path=str(path))
col = client.create_collection(
    args.collection, embedding_function=None,
    configuration={"hnsw": {"space": "cosine", "max_neighbors": args.m,
                            "ef_construction": args.ef_construct, "ef_search": args.ef_search}})
for i in range(0, args.n, args.batch):
    part = payloads[i:i + args.batch]
    col.add(ids=[str(p["doc_id"]) for p in part], embeddings=vectors[i:i + args.batch],
            documents=[p["text"] for p in part],
            metadatas=[{k: v for k, v in p.items() if k != "text"} for p in part])
t_insert = time.perf_counter() - t0
count = col.count()
stop.set()
sampler.join()
rss_after = proc.memory_info().rss / 2**20
del col, client

apparent = allocated = 0
for f in path.rglob("*"):
    if f.is_file():
        st = f.stat()
        apparent += st.st_size
        allocated += st.st_blocks * 512

print(json.dumps({"count": count, "insert_s": t_insert, "rss_before_mb": rss_before, "rss_after_mb": rss_after,
                  "rss_peak_mb": peak["rss"], "disk_apparent_mb": apparent / 2**20,
                  "disk_allocated_mb": allocated / 2**20}))
