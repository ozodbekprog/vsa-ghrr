#!/usr/bin/env python3
"""Chunked, restart-safe downloader (stdlib only).

Each file is preallocated to its Content-Length and fetched in 8 MiB ranges by a
small thread pool. Completed chunk indices are appended to <file>.done so a
restart never re-downloads work. Flaky links are the norm here: every chunk is
retried individually with a bounded backoff.
"""
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

CHUNK = 4 << 20
WORKERS = 6
TRIES = 300


def head_len(url: str, tries: int = 100) -> int:
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=60) as r:
                n = int(r.headers.get("Content-Length", 0))
                if n:
                    return n
        except Exception:
            pass
        time.sleep(min(10.0, 1.0 + attempt))
    return 0


def load_done(path: str) -> set[int]:
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {int(x) for x in f.read().split() if x.strip()}


def fetch_chunk(url: str, path: str, idx: int, start: int, end: int) -> bool:
    n = end - start + 1
    for attempt in range(TRIES):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if len(data) == n:
                with open(path, "r+b") as f:
                    f.seek(start)
                    f.write(data)
                return True
        except Exception:
            pass
        time.sleep(min(3.0, 0.3 * (attempt + 1)))
    return False


def download(url: str, dest: str) -> bool:
    name = os.path.basename(url.split("?")[0])
    path = os.path.join(dest, name)
    done_path = path + ".done"
    want = head_len(url)
    if not want:
        print(f"SKIP {name}: no content-length (dns?)", flush=True)
        return False
    if os.path.exists(path) and os.path.getsize(path) == want and len(load_done(done_path)):
        print(f"skip {name}", flush=True)
        return True
    if not os.path.exists(path) or os.path.getsize(path) != want:
        with open(path, "wb") as f:
            f.truncate(want)
    ranges = [(i, s, min(s + CHUNK - 1, want - 1))
              for i, s in enumerate(range(0, want, CHUNK))]
    done = load_done(done_path)
    todo = [r for r in ranges if r[0] not in done]
    print(f"start {name}: {len(todo)}/{len(ranges)} chunks left", flush=True)

    lock_file = open(done_path, "a")

    def run(item):
        idx, a, b = item
        ok = fetch_chunk(url, path, idx, a, b)
        if ok:
            lock_file.write(f"{idx}\n")
            lock_file.flush()
        return ok

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        results = list(ex.map(run, todo))
    lock_file.close()
    good = all(results) and len(load_done(done_path)) == len(ranges)
    dt = time.time() - t0
    print(("ok " if good else "PARTIAL ") + f"{name} in {dt:.0f}s", flush=True)
    return good


def main():
    urls = [l.strip() for l in open(sys.argv[1]) if l.strip()]
    dest = sys.argv[2]
    os.makedirs(dest, exist_ok=True)
    for u in urls:
        try:
            download(u, dest)
        except Exception as e:
            print(f"ERROR {u}: {type(e).__name__} {e}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
