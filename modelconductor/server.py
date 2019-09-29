__package__ = "modelconductor"
# !/usr/bin/env python3
"""Server for multithreaded (asynchronous) chat application."""
from socket import AF_INET, socket, SOCK_STREAM
from threading import Thread

_EVENT = None
_QUEUE = None
_SERVER = None


def accept_incoming_connections():
    """Sets up handling for incoming clients."""
    while True:
        client, client_address = _SERVER.accept()
        print("%s:%s has connected." % client_address)
        addresses[client] = client_address
        t = Thread(target=handle_client, args=(client,), daemon=True)
        t.start()


def handle_client(client):  # Takes client socket as argument.
    """Handles a single client connection."""
    while True:
        headers = b''
        msg = b''
        # make sure header is received completely
        while len(headers) < HEADERSIZE:
            headers += client.recv(HEADERSIZE)

        # handle overshoot
        msg += headers[HEADERSIZE:]
        headers = headers[:HEADERSIZE]

        # determine length of stuff to be received
        msglen = int(headers)

        while True:
            receive_at_most = msglen - len(msg)
            if receive_at_most == 0:
                break
            msg += client.recv(receive_at_most)

        print("Received message: ", msg[0:50])
        _QUEUE.put(msg)


clients = {}
addresses = {}

HOST = ''
PORT = 33003
BUFSIZ = 64
ADDR = (HOST, PORT)
HEADERSIZE = 10


def run(event, queue):
    global _EVENT
    global _QUEUE
    global _SERVER
    _EVENT = event
    _QUEUE = queue
    _SERVER = socket(AF_INET, SOCK_STREAM)
    _SERVER.bind(ADDR)
    _SERVER.listen(5)
    print("Waiting for connection...")
    ACCEPT_THREAD = Thread(target=accept_incoming_connections, daemon=True)
    ACCEPT_THREAD.start()
    ACCEPT_THREAD.join()
    _SERVER.close()
