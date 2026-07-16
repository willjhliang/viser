"""Smoke tests for the HTTP request handler in the websocket server.

Exercises the request path that serves the client bundle and performs path
traversal validation. Regression guard for cases where Python version
incompatibilities (e.g. methods that only exist on newer Pythons) would raise
inside ``process_request`` and surface as a 500 with the websockets library's
default failure message."""

import gzip
import hashlib
import http.client
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Optional, Tuple
from unittest.mock import patch

import viser
import viser._client_autobuild
from viser import infra


def _raw_get_status(host: str, port: int, raw_target: str) -> int:
    """Send GET with a literal request-line target so percent-escapes
    aren't normalized client-side."""
    conn = http.client.HTTPConnection(host, port, timeout=5)
    try:
        conn.request("GET", raw_target)
        return conn.getresponse().status
    finally:
        conn.close()


def _raw_get(
    host: str,
    port: int,
    raw_target: str,
    headers: Optional[Dict[str, str]] = None,
) -> Tuple[int, Dict[str, str], bytes]:
    conn = http.client.HTTPConnection(host, port, timeout=5)
    try:
        conn.request("GET", raw_target, headers=headers or {})
        response = conn.getresponse()
        response_headers = {key.lower(): value for key, value in response.getheaders()}
        return response.status, response_headers, response.read()
    finally:
        conn.close()


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@patch.object(viser._client_autobuild, "ensure_client_is_built", lambda: None)
def test_http_root_and_traversal():
    server = viser.ViserServer()
    port = server.get_port()
    try:
        # Give the websockets server a moment to accept connections.
        time.sleep(0.1)

        # A normal asset fetch must reach the static file lookup without
        # raising. We don't require 200 here (the client bundle may not be
        # present in every test environment), but any exception in
        # process_request turns into a 500 from the websockets library.
        try:
            status = urllib.request.urlopen(
                f"http://localhost:{port}/", timeout=5
            ).status
        except urllib.error.HTTPError as e:
            status = e.code
        assert status != 500, "process_request raised for GET /"

        # Path traversal must be rejected as 404, not leak out of the client
        # root.
        try:
            status = urllib.request.urlopen(
                f"http://localhost:{port}/../../../etc/passwd", timeout=5
            ).status
        except urllib.error.HTTPError as e:
            status = e.code
        assert status == 404
    finally:
        server.stop()


def _fetch(url: str) -> Tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, b""


def test_http_etag_revalidation(tmp_path: Path) -> None:
    served_root = tmp_path / "served"
    served_root.mkdir()
    payload = b"<html>cache me</html>" * 100
    (served_root / "index.html").write_bytes(payload)

    server = infra.WebsockServer(
        host="127.0.0.1",
        port=_find_free_port(),
        http_server_root=served_root,
        verbose=False,
    )
    server.start()
    try:
        time.sleep(0.1)
        # start() may have fallen back to a nearby port if ours was taken.
        port = server._port
        status, headers, body = _raw_get(
            "127.0.0.1", port, "/", {"Accept-Encoding": "identity"}
        )
        assert status == 200
        assert body == payload
        identity_etag = headers["etag"]
        assert identity_etag.startswith('"') and identity_etag.endswith('"')
        assert headers["cache-control"] == "no-cache"
        assert headers["vary"] == "Accept-Encoding"

        status, headers, body = _raw_get(
            "127.0.0.1",
            port,
            "/",
            {"Accept-Encoding": "identity", "If-None-Match": identity_etag},
        )
        assert status == 304
        assert body == b""
        assert headers["etag"] == identity_etag
        assert "content-length" not in headers

        status, _, _ = _raw_get(
            "127.0.0.1",
            port,
            "/",
            {"If-None-Match": f'"other", W/{identity_etag}'},
        )
        assert status == 304
        assert _raw_get("127.0.0.1", port, "/", {"If-None-Match": "*"})[0] == 304

        status, headers, body = _raw_get(
            "127.0.0.1", port, "/", {"Accept-Encoding": "gzip"}
        )
        assert status == 200
        assert gzip.decompress(body) == payload
        gzip_etag = headers["etag"]
        assert gzip_etag != identity_etag

        # ETags must be deterministic across server restarts, or every
        # restart would invalidate all client caches. For gzip this depends
        # on compressing with mtime=0; recompute both tags independently.
        assert identity_etag == f'"{hashlib.sha256(payload).hexdigest()}"'
        expected_gzip = hashlib.sha256(gzip.compress(payload, mtime=0)).hexdigest()
        assert gzip_etag == f'"{expected_gzip}"'

        assert (
            _raw_get(
                "127.0.0.1",
                port,
                "/",
                {"Accept-Encoding": "gzip", "If-None-Match": identity_etag},
            )[0]
            == 200
        )
        assert (
            _raw_get(
                "127.0.0.1",
                port,
                "/",
                {"Accept-Encoding": "gzip", "If-None-Match": gzip_etag},
            )[0]
            == 304
        )
    finally:
        server.stop()


def test_http_serves_files_through_symlink(tmp_path: Path):
    # Regression test for Bazel/uv runfile layouts where the served root and
    # the files inside it pass through different symlinks. With the old
    # Path.resolve() based check, the resolved file path lands outside the
    # resolved root and every static asset returns 404.
    served_root = tmp_path / "served"
    served_root.mkdir()
    real_target_dir = tmp_path / "elsewhere"
    real_target_dir.mkdir()

    real_index = real_target_dir / "index.html"
    real_index.write_bytes(b"<html>hello from symlinked file</html>")
    (served_root / "index.html").symlink_to(real_index)

    real_asset = real_target_dir / "asset.js"
    real_asset.write_bytes(b"console.log('via symlink');")
    (served_root / "asset.js").symlink_to(real_asset)

    # A real file one level above served_root. Any traversal that
    # successfully escapes would resolve to this file and return its
    # contents -- so a 200 here is a clean indicator that the check
    # leaks, even on platforms where percent-encoded names happen to
    # not exist on disk.
    secret_file = tmp_path / "secret.txt"
    secret_file.write_bytes(b"SECRET")

    server = infra.WebsockServer(
        host="127.0.0.1",
        port=18900,
        http_server_root=served_root,
        verbose=False,
    )
    server.start()
    try:
        time.sleep(0.1)
        port = server._port

        status, body = _fetch(f"http://127.0.0.1:{port}/")
        assert status == 200
        assert b"hello from symlinked file" in body

        status, body = _fetch(f"http://127.0.0.1:{port}/asset.js")
        assert status == 200
        assert b"console.log('via symlink');" in body

        # Traversal still rejected even though we no longer use resolve().
        status, body = _fetch(f"http://127.0.0.1:{port}/../secret.txt")
        assert status == 404
        assert b"SECRET" not in body

        # Percent-encoded traversal must also be rejected -- the check
        # has to URL-decode before splitting URL segments, otherwise
        # ``%2e%2e`` could slip through.
        assert _raw_get_status("127.0.0.1", port, "/%2e%2e/secret.txt") == 404
        assert (
            _raw_get_status("127.0.0.1", port, "/foo/%2e%2e/%2e%2e/secret.txt") == 404
        )

        # Backslash-encoded traversal must also be rejected. ``pathlib``
        # on Linux treats backslashes as literal filename characters,
        # which would let ``foo\..\bar`` skip a parts-only check.
        assert _raw_get_status("127.0.0.1", port, "/foo\\..\\..\\secret.txt") == 404

        # Missing file still 404s.
        status, _ = _fetch(f"http://127.0.0.1:{port}/does-not-exist.js")
        assert status == 404

        # A request that resolves to a real *directory* must 404, not raise
        # IsADirectoryError inside read_bytes() (which surfaces as a 500).
        (served_root / "subdir").mkdir()
        assert _fetch(f"http://127.0.0.1:{port}/subdir")[0] == 404
        assert _fetch(f"http://127.0.0.1:{port}/subdir/")[0] == 404
    finally:
        server.stop()
