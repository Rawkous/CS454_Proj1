#!/bin/sh
# CS 454 - Project 1 - launcher for the department server.
#
# The program is Python 3; this script just invokes it with unbuffered I/O
# so progress messages for the L'' sweep appear promptly.  Pass -t to run
# the independent verification harness instead of the interactive menu.
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"

case "${1-}" in
    -t|--test|--verify)
        exec python3 -u "$HERE/verify.py"
        ;;
    *)
        exec python3 -u "$HERE/project1.py"
        ;;
esac
