#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS 454 - Project 1 - Independent verification harness.

This file is NOT part of the graded program.  It cross-checks the main
project1.py in three independent ways so we can trust the n=300 answer:

  Check 1 (small L)      - brute force all 4^n strings for n <= 11 and
                           compare counts against the DFA for L.
  Check 2 (small L')     - brute force all 4^n strings for n <= 12 and
                           compare against the L'' DFA.
  Check 3 (medium L')    - independent forward DP on the single L DFA
                           with the two middle symbols forced to 'a' for
                           n = 2..22, compared against the L'' DFA.
  Check 4 (large L')     - factored path-count cross-check at n = 300.
                           Because delta''((p,q),(x,y)) = (dL(p,x), dL(q,y))
                           the two tracks of L'' are independent at the
                           path-counting level, so
                              N''_0(k) = sum_{p,q} F(p,k) * F(q,k) * accept(p,q)
                           where F(p,k) is the number of length-k words
                           that drive the L DFA from state 0 to state p.
                           We compute F via a single forward sweep of the
                           L DFA (fast) and then evaluate the double sum
                           in numpy.

Run:
    $ python3 verify.py
"""

from __future__ import annotations

import itertools
import sys
import time
from typing import List, Sequence

import numpy as np

import project1 as P


# ---------------------------------------------------------------------------
# Brute-force reference implementations
# ---------------------------------------------------------------------------

def _in_L(w: Sequence[str]) -> bool:
    """True iff every length-6 substring of w contains a,b,c,d."""
    for i in range(len(w) - 5):
        s = set(w[i:i + 6])
        if not ('a' in s and 'b' in s and 'c' in s and 'd' in s):
            return False
    return True


def brute_count_L(n: int) -> int:
    """Enumerate all 4^n strings and count those in L."""
    return sum(1 for w in itertools.product('abcd', repeat=n) if _in_L(w))


def brute_count_Lprime(n: int) -> int:
    """Enumerate all 4^n strings and count those in L' (even n, middle 'aa')."""
    if n % 2 == 1:
        return 0
    m = n // 2
    return sum(
        1
        for w in itertools.product('abcd', repeat=n)
        if w[m - 1] == 'a' and w[m] == 'a' and _in_L(w)
    )


# ---------------------------------------------------------------------------
# Independent forward DP for L' using only the single L DFA
# ---------------------------------------------------------------------------

def forward_dp_count_Lprime(
    delta_L: Sequence[Sequence[int]],
    accepting_L: Sequence[bool],
    n: int,
) -> int:
    """Count length-n L' strings by running a forward DP on the single L DFA,
    forcing the two middle positions to be 'a'.

    This is independent of the product construction in project1.py: it uses
    only the forward L DFA and conditioning on the two middle symbols.
    """
    if n % 2 == 1:
        return 0
    m = n // 2
    N = len(delta_L)
    f = [0] * N
    f[0] = 1                                 # start state, before reading anything
    for i in range(n):
        g = [0] * N
        syms = (0,) if i in (m - 1, m) else (0, 1, 2, 3)   # 0 = 'a'
        for j in range(N):
            if f[j]:
                dj = delta_L[j]
                v = f[j]
                for x in syms:
                    g[dj[x]] += v
        f = g
    # Count paths ending at any accepting state (= any non-fail state).
    return sum(f[j] for j in range(N) if accepting_L[j])


# ---------------------------------------------------------------------------
# Factored product cross-check for large n
# ---------------------------------------------------------------------------

def factored_count_Lprime(
    delta_L: Sequence[Sequence[int]],
    accepting_L: Sequence[bool],
    n: int,
) -> int:
    """Count length-n L' strings via the factored path-count identity

           N''_0(k) = sum_{p,q} F(p,k) F(q,k) accept(p,q)

    where F(p,k) is the number of length-k words driving the L DFA from 0 to p.
    Fast: one 1366-wide forward sweep plus one (1366,1366) numpy expression.
    """
    if n % 2 == 1:
        return 0
    m = n // 2
    k = m - 1
    N = len(delta_L)

    # Forward sweep: F_k[p] = number of length-k words driving L DFA 0 -> p.
    F = [0] * N
    F[0] = 1
    for _ in range(k):
        G = [0] * N
        for j in range(N):
            if F[j]:
                dj = delta_L[j]
                v = F[j]
                G[dj[0]] += v; G[dj[1]] += v; G[dj[2]] += v; G[dj[3]] += v
        F = G

    # Rebuild the acceptance mask (reshape to (M_L, M_L) for the double sum).
    accept = P._acceptance_mask_Lpp(accepting_L).reshape(P.M_L, P.M_L)

    F_arr = np.empty(N, dtype=object)
    F_arr[:] = F
    outer = F_arr[:, None] * F_arr[None, :]               # (M_L, M_L) big-ints
    total = int(outer[accept].sum())
    return total


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def main() -> None:
    print("Building L DFA ...")
    dL, aL = P.build_L_dfa()
    assert len(dL) == P.M_L
    print("  L has %d states (expected %d)" % (len(dL), P.M_L))

    print("\n----- Check 1: L vs brute force for n = 0..11 -----")
    N0_L = P.count_strings_L(dL, aL, 11)
    all_ok = True
    for n in range(12):
        bf = brute_count_L(n)
        tag = "OK" if N0_L[n] == bf else "MISMATCH"
        if N0_L[n] != bf:
            all_ok = False
        print("  n=%2d  DFA=%d  brute=%d  %s" % (n, N0_L[n], bf, tag))
    assert all_ok, "L vs brute force disagreed"

    print("\n----- Check 2: L' vs brute force for even n in 2..12 -----")
    # Build L'' lazily; this takes a few seconds but is unavoidable.
    cpp = P._LppCounter()
    for n in range(2, 13, 2):
        k = n // 2 - 1
        dfa_val = cpp.get(k)
        bf = brute_count_Lprime(n)
        tag = "OK" if dfa_val == bf else "MISMATCH"
        if dfa_val != bf:
            all_ok = False
        print("  n=%2d  L''(k=%d)=%d  brute=%d  %s" % (n, k, dfa_val, bf, tag))
    assert all_ok, "L' vs brute force disagreed"

    print("\n----- Check 3: L' vs forward-DP for even n in 2..22 -----")
    for n in range(2, 23, 2):
        k = n // 2 - 1
        dfa_val = cpp.get(k)
        fp = forward_dp_count_Lprime(dL, aL, n)
        tag = "OK" if dfa_val == fp else "MISMATCH"
        if dfa_val != fp:
            all_ok = False
        print("  n=%2d  L''(k=%d)=%d  fwd-DP=%d  %s" % (n, k, dfa_val, fp, tag))
    assert all_ok, "L' vs forward DP disagreed"

    print("\n----- Check 4: factored cross-check at n = 300 -----")
    t0 = time.time()
    factored = factored_count_Lprime(dL, aL, 300)
    print("  factored method (n=300): %.1fs" % (time.time() - t0))
    print("  factored result = %d" % factored)

    # Now get the main L'' counter's answer for n=300 (this is the slow one).
    t0 = time.time()
    main_val = cpp.get(149)                   # n=300 -> k=149
    print("  L'' counter (n=300):     %.1fs" % (time.time() - t0))
    tag = "OK" if main_val == factored else "MISMATCH"
    if main_val != factored:
        all_ok = False
    print("  equal? %s" % tag)

    print("\n%s" % ("ALL CHECKS PASSED." if all_ok else "SOME CHECKS FAILED."))
    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
