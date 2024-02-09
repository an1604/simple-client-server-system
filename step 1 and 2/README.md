# Simple Client-Server System

This is a basic client-server system where the server executes commands sent by the client and returns responses accordingly.

## Server

The server script (`server.py`) listens for connections from clients on a specified host and port. It supports the following commands:

- **TIME**: Returns the current time.
- **WHORU**: Returns a string representing the server's name.
- **RAND**: Returns a random number between 1 and 10.
- **EXIT**: Disconnects from the client.

The server keeps receiving requests from the client until it receives an EXIT command.

## Client

The client script (`client.py`) allows users to send commands to the server. It prompts the user to select one of the supported commands mentioned above. The client then sends the request to the server, receives the response, and displays it to the user.

## Protocol

The client and server communicate using a simple protocol that allows messages of variable length. Each message is prefixed with its length, allowing the receiving party to know how much data to expect. The protocol ensures that both parties can handle messages of any length without issues.

### Level 1

The client and server implement a basic communication protocol where the length of each message is fixed to 1024 bytes.

### Level 2

The client and server enhance the protocol to support messages of variable length. They coordinate in advance on the number of digits used to represent the message length and pad the length with zeros if necessary. This ensures that both parties can accurately determine the length of incoming messages.

## Instructions

1. Run the server script (`server.py`) to start the server.
2. Run the client script (`client.py`) to connect to the server and send commands.
3. Follow the prompts in the client script to interact with the server.

## Notes

- Make sure the server is running before connecting with the client.
- Ensure that the host and port settings in both client and server scripts match.
- You can customize the server responses and add new commands as needed.
