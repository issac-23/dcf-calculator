"""Block outbound network access and remember every attempt.

Raising is not enough on its own: dcf/data.py catches bare Exception, so a
blocked connection can be swallowed and the test still goes green. That is how
a Treasury fetch once cost 30 seconds a run and nothing failed. Every blocked
attempt is therefore recorded, and the conftest fixture fails the test at
teardown if the record is not empty.

Two layers, because yfinance 1.x talks to Yahoo through curl_cffi (libcurl),
which never touches Python's socket module:

* socket: DNS lookups and connects, which catches urllib, http.client, requests.
* curl_cffi: Session.request and AsyncSession.request, which catches yfinance.

Loopback and unix sockets stay open for Streamlit's AppTest.

If curl_cffi is ever absent (a future yfinance without it), that layer is
skipped and the socket layer still covers requests and urllib. Known limit:
code that binds socket functions at import time (`from socket import
getaddrinfo`) keeps the real function and bypasses the guard.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlsplit

import pytest

try:
    import curl_cffi.requests as curl_requests
except ImportError:
    curl_requests = None

_UNIX = "unix"  # stands in for AF_UNIX peers, which have no host


def _is_local(host: str | None) -> bool:
    """Loopback in any spelling, the unspecified address, and the empty host."""
    if host is None:
        return False
    if host == "" or host == _UNIX or host.lower() == "localhost":
        return True
    try:
        ip = ipaddress.ip_address(host.split("%")[0])
    except ValueError:
        return False  # a real hostname
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return ip.is_loopback or ip.is_unspecified


class NetworkBlocked(OSError):
    """Raised when a test tries to reach a host outside the loopback."""


class NetworkGuard:
    def __init__(self) -> None:
        self.blocked: list[str] = []

    def _check(self, destination: str, host: str | None) -> None:
        if _is_local(host):
            return
        self.blocked.append(destination)
        raise NetworkBlocked(f"network blocked in tests: {destination}")

    def install(self, monkeypatch) -> None:
        real_connect = socket.socket.connect
        real_connect_ex = socket.socket.connect_ex
        real_getaddrinfo = socket.getaddrinfo

        def host_of(sock, address):
            # AF_UNIX addresses are paths, not (host, port) tuples.
            if sock.family == getattr(socket, "AF_UNIX", None):
                return _UNIX
            return address[0] if isinstance(address, tuple) else _UNIX

        def connect(sock, address):
            self._check(f"socket connect {address!r}", host_of(sock, address))
            return real_connect(sock, address)

        def connect_ex(sock, address):
            self._check(f"socket connect {address!r}", host_of(sock, address))
            return real_connect_ex(sock, address)

        def getaddrinfo(host, *args, **kwargs):
            if isinstance(host, bytes):
                host = host.decode()
            if host is not None:  # None and "" are passive/local lookups
                self._check(f"dns lookup {host!r}", host)
            return real_getaddrinfo(host, *args, **kwargs)

        monkeypatch.setattr(socket.socket, "connect", connect)
        monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
        monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)

        def guarded_request(real):
            def request(session, method, url, *args, **kwargs):
                self._check(f"{method} {url}", urlsplit(str(url)).hostname)
                return real(session, method, url, *args, **kwargs)

            return request

        if curl_requests is None:
            return
        for cls in (curl_requests.Session, curl_requests.AsyncSession):
            monkeypatch.setattr(cls, "request", guarded_request(cls.request))


@pytest.fixture(autouse=True)
def network_guard(request, monkeypatch):
    """Fail any test that tries to leave the machine, unless it is marked live.

    Raising alone is not enough: dcf/data.py swallows exceptions, so a blocked
    call can vanish and the test stay green. The guard records each attempt and
    this fixture fails the test at teardown if there was one, whether or not
    the exception surfaced. See tests/offline_guard.py.
    """
    guard = NetworkGuard()
    if request.node.get_closest_marker("live"):
        yield guard
        return
    guard.install(monkeypatch)
    yield guard
    if guard.blocked:
        attempts = ", ".join(guard.blocked)
        pytest.fail(f"test tried to reach the network: {attempts}", pytrace=False)
