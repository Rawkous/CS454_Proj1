#!/bin/sh
# CS 454 - Project 1 - launcher for the department server.
# B. Axel Martinez and Emilio Orozco  (Fall 2026)
# We wrote this launcher together alongside the rest of the project,
# coordinating remotely over Discord and the shared repo.
#
# Our program is Python 3; this script just invokes it with unbuffered I/O
# so our progress messages for the L'' sweep appear promptly.  Pass -t to
# run our independent verification harness instead of the interactive menu.
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
