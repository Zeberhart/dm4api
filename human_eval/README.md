Human Study
===========

We evaluated two variations of the dialogue manager with 4 human study participants. Each participant used a search tool implementinng the dialogue manager to find functions in the Libssh API. Two participants were given the baseline search policy for the Libssh tasks, and two were given a learned policy.

## Interface
The API search tool created for programmers relies on simple templated system messages and user quick responses. The interface itself is implemented in Javascript with React, and it connects to a backend server running the dialogue manager written in Python 3 using the Sanic server framework.  In order to run the tool, please follow these procedures:

1. Navigate to the /backend folder, and start the server by running `python server.py`

2. In another terminal, avigate to the /frontend folder. Install the required node package by running  `npm install`

3. Start the frontend server by running `npm run dev`

4. On the same machine, open a web browser and navigate to `http://localhost:3000/`


## Search Tasks
We gave programmers the following search tasks for the libssh library in the human study:

1. Create a new file on a remote server using SSH File Transfer Protocol.
    * Answer: sftp_open
2. Determine whether or not you can authenticate a user to a server with a given public key, but do not actually perform any authentication.
    * Answer: ssh_userauth_try_publickey
3. Make a new folder with a given name on a remote server, using SCP protocol.
    * Answer: ssh_scp_push_directory
4. You have the ssh_poll_handle for a poll object. Associate a function with the poll object such that that function will be called whenever the object is polled.
    * Answer: ssh_poll_set_callback
5. Before connecting to an SSH server, specify a host and port for the ssh_session.
    * Answer: ssh_options_set
6. Initiate the process of receiving a file over scp. This should not actually receive any file data.
    * Answer: ssh_scp_pull_request

The questions were worded such that, if the participants were to search for an entire question string, the true answer would not be included in the first set of results.
