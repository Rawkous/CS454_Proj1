#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS 454 - Theory of Computation - Project 1
B. Axel Martinez and Emilio Orozco  (Fall 2026)

Count the number of strings w of length n over the alphabet {a,b,c,d} such
that every length-6 substring of w contains all four letters.  Call this
language L.  Also count the strings of even length n in L whose two middle
symbols are 'aa' (language L').

Both counts are produced by building the appropriate DFA explicitly and
running the two-array N_j(k) recurrence described in the project handout:

    N_j(k)  =  sum_{x in Sigma}  N_{delta(j,x)}(k-1)
    N_j(0)  =  1 if j is accepting, else 0

with the final answer being N_0(n), using two length-|Q| arrays "prev" and
"next" and copying "next" back into "prev" after each iteration.

Two DFAs are built:

    L     : 1366 states, 4 symbols         (shift register of last 5 symbols)
    L''   : 1366^2 = 1,865,956 states, 16 symbols  (product construction;
            see the long comment above build_Lpp_dfa for the derivation)

The number of length-n strings in L' is N''_0(n/2 - 1) for even n, 0 for odd n.

Run the file directly for the interactive menu described in the handout:

    $ python3 project1.py
"""

from __future__ import annotations

import sys
import time
from typing import Callable, List, Sequence, Tuple

import numpy as np


# =============================================================================
# Section 1 - Alphabet and base-4 state encoding (follows the PDF exactly)
# =============================================================================

ALPHABET: Tuple[str, ...] = ('a', 'b', 'c', 'd')

# PDF digit assignment: a=1, b=2, c=3, d=4 so that the all-'a' buffer does not
# collide with a shorter buffer.  Example from the handout:
#   babca -> 2*4^4 + 1*4^3 + 2*4^2 + 3*4^1 + 1*4^0 = 512+64+32+12+1 = 621
DIGIT = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
INV_DIGIT = {v: k for k, v in DIGIT.items()}

# Buffers have length 0..5, giving 1+4+16+64+256+1024 = 1365 valid states.
# The additional "fail" state (reached once a bad length-6 window is scanned)
# is labelled 1365.  Hence the L DFA has M_L = 1366 states.
M_L: int = 1366
FAIL_L: int = 1365


def encode_buffer(buf: Sequence[str]) -> int:
    """Return the base-4 integer code of ``buf`` per the PDF scheme.

    Inputs
    ------
    buf : sequence of characters, each in {'a','b','c','d'}, with length 0..5.
          The empty sequence represents the start state.

    Output
    ------
    An integer in 0..1364 that uniquely identifies ``buf``.

    Example
    -------
        encode_buffer(())          == 0
        encode_buffer(('a',))      == 1
        encode_buffer(('b','a','b','c','a')) == 621    # the PDF's example

    Pre-conditions
    --------------
        - every character in ``buf`` is one of 'a','b','c','d'
        - len(buf) <= 5
    """
    code = 0
    for ch in buf:
        code = code * 4 + DIGIT[ch]
    return code


def _enumerate_all_buffers() -> List[Tuple[str, ...]]:
    """Return all buffers of length 0..5 over the alphabet (helper).

    Output
    ------
    A list of 1365 tuples: () ; ('a',) ... ('d',) ; ('a','a') ... ; ...
    """
    buffers: List[Tuple[str, ...]] = []
    frontier: List[Tuple[str, ...]] = [()]
    buffers.append(())
    for _ in range(5):
        new_frontier = []
        for buf in frontier:
            for ch in ALPHABET:
                nb = buf + (ch,)
                new_frontier.append(nb)
                buffers.append(nb)
        frontier = new_frontier
    return buffers


def _build_buffer_tables() -> Tuple[List[Tuple[str, ...]], dict]:
    """Build code<->buffer lookup tables.  Asserts the encoding is a bijection
    onto 0..1364.
    """
    buffers = _enumerate_all_buffers()
    code_of: dict = {}
    for buf in buffers:
        c = encode_buffer(buf)
        if c in code_of:
            raise AssertionError(
                "encoding collision: %r and %r both map to %d"
                % (code_of[c], buf, c)
            )
        code_of[c] = buf
    assert len(code_of) == 1365, "expected 1365 valid buffers, got %d" % len(code_of)
    assert sorted(code_of) == list(range(1365)), (
        "codes are not a contiguous 0..1364 range"
    )
    return buffers, code_of


# Decode table (code -> buffer tuple); built once at import time.
_BUFFERS, _CODE_OF_CODE = _build_buffer_tables()


def decode_state(code: int) -> Tuple[str, ...]:
    """Inverse of :func:`encode_buffer`.  Returns the buffer at state ``code``.

    Inputs
    ------
    code : an integer in 0..1364 (a valid buffer state; the fail state 1365
           has no buffer and raises).

    Output
    ------
    A tuple of 0..5 characters.

    Example
    -------
        decode_state(0)   == ()
        decode_state(621) == ('b','a','b','c','a')

    Pre-conditions
    --------------
        - 0 <= code <= 1364
    """
    if not (0 <= code <= 1364):
        raise ValueError("decode_state: code %d is not a valid buffer code" % code)
    return _CODE_OF_CODE[code]


# =============================================================================
# Section 2 - DFA for L
# =============================================================================


def _window_ok(w: Sequence[str]) -> bool:
    """True iff the length-6 window ``w`` contains all four letters a,b,c,d."""
    s = set(w)
    return 'a' in s and 'b' in s and 'c' in s and 'd' in s


def build_L_dfa() -> Tuple[List[List[int]], List[bool]]:
    """Build the DFA for the language L.

    L = { w over {a,b,c,d} : every length-6 substring of w contains a,b,c,d }

    The DFA has 1366 states (labelled 0..1365 per :func:`encode_buffer`,
    with 1365 = fail) and 4 input symbols (indices 0..3 for a,b,c,d).

    Transition rule (from the handout)
    ----------------------------------
    If state ``q`` encodes a buffer ``b1..bl`` with l < 5, then delta(q, x)
    encodes ``b1..bl x``.  If l == 5 then delta(q, x) is the fail state if
    ``b1..b5 x`` is missing any of a,b,c,d, else it encodes ``b2..b5 x``.
    The fail state loops to itself on every symbol.

    Output
    ------
    (delta, accepting) where
      - delta[j][x] is the next state from j on reading symbol index x
      - accepting[j] is True iff j is an accepting state (all states except
        the fail state are accepting: strings shorter than 6 are vacuously
        in L, and any string that has avoided the fail state so far is a
        prefix extendible to a word in L).

    Example
    -------
        delta, accepting = build_L_dfa()
        delta[0][0] == encode_buffer(('a',))      # from start on 'a'
        accepting[FAIL_L] is False
    """
    delta: List[List[int]] = [[0] * 4 for _ in range(M_L)]
    accepting: List[bool] = [True] * M_L
    accepting[FAIL_L] = False

    for buf in _BUFFERS:
        j = encode_buffer(buf)
        for x_idx, ch in enumerate(ALPHABET):
            if len(buf) < 5:
                delta[j][x_idx] = encode_buffer(buf + (ch,))
            else:
                window = buf + (ch,)            # length 6
                if _window_ok(window):
                    delta[j][x_idx] = encode_buffer(buf[1:] + (ch,))
                else:
                    delta[j][x_idx] = FAIL_L

    # Fail state loops to itself.
    for x_idx in range(4):
        delta[FAIL_L][x_idx] = FAIL_L

    return delta, accepting


# =============================================================================
# Section 3 - DFA for L''
# =============================================================================
#
# Derivation (the piece the handout defers to class)
# --------------------------------------------------
# A word in L' of length n = 2m has the form   w = u 'a' 'a' v   with
# |u| = |v| = m - 1, and w must be in L.  Reading u left-to-right and v
# right-to-left in lockstep yields a word   z = (u_1, v_{m-1})
# (u_2, v_{m-2}) ... (u_{m-1}, v_1)  of length m-1 over the 16-symbol pair
# alphabet Sigma x Sigma.
#
# A product state (p, q) tracks the forward L-DFA state on u (component p)
# and the forward L-DFA state on reversed-v (component q); the forward
# L-DFA on reversed-v coincides with the "backward" DFA on v because the
# window predicate is symmetric under reversal.  Each product state has
# 16 outgoing transitions:
#
#     delta''((p,q), (x,y)) = (delta_L(p, x), delta_L(q, y))
#
# The start state is (0, 0) = (epsilon, epsilon).  A product state (p, q)
# is accepting iff neither component is the fail state AND the joined
# string   buf(p) + "aa" + reverse(buf(q))   has every length-6 window
# containing a,b,c,d.  Each junction-crossing window has at most 5 symbols
# on either side of "aa", so the joined length-<=12 string captures every
# window that straddles the middle; windows fully inside u or v were
# already policed by the component DFAs.
#
# Hence the count of length-n strings in L' (n = 2m, even) is
# N''_0(m-1), and 0 for odd n.
#
# State labelling for L'' per the handout: state (p, q) carries integer
# label p * M_L + q, so the labels run 0 .. M_L^2 - 1 = 1,865,955.  Input
# symbols are labelled 4*x + y in 0..15 where x, y are 0..3 digits.


M_LPP: int = M_L * M_L           # 1,865,956 states


def _acceptance_mask_Lpp(accepting_L: Sequence[bool]) -> np.ndarray:
    """Return a boolean array of length M_LPP: True iff state is accepting.

    The acceptance test loops only over the 1365 x 1365 non-fail product
    states (the fail-involving states are unconditionally non-accepting),
    which keeps the construction fast.
    """
    accept = np.zeros(M_LPP, dtype=bool)
    aa = ('a', 'a')
    for p in range(M_L):
        if not accepting_L[p]:
            continue                                   # p is the fail state
        bp = _CODE_OF_CODE[p]                          # buffer for p
        for q in range(M_L):
            if not accepting_L[q]:
                continue                               # q is the fail state
            bq = _CODE_OF_CODE[q]                      # buffer for q
            joined = bp + aa + bq[::-1]
            # Check every length-6 window of the joined string.
            ok = True
            for i in range(len(joined) - 5):
                if not _window_ok(joined[i:i + 6]):
                    ok = False
                    break
            if ok:
                accept[p * M_L + q] = True
    return accept


def build_Lpp_dfa(
    delta_L: Sequence[Sequence[int]],
    accepting_L: Sequence[bool],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the DFA for L'' (the product construction described above).

    Inputs
    ------
    delta_L     : transitions of the L DFA, delta_L[j][x] in 0..M_L-1
    accepting_L : accepting flags of the L DFA, length M_L

    Outputs
    -------
    (delta_cols, delta_table, accepting) where
      - delta_cols is an int32 array of shape (M_L, 4) holding delta_L in
        numpy form - this is the "factored" view used by the two-stage
        iteration (it is literally delta_L).
      - delta_table is an int32 array of shape (M_LPP, 16) holding the
        full product transition function so the DFA is materialised
        explicitly, as the handout requires.  Row index is p*M_L + q;
        column index is 4*x + y.  Each entry equals
        delta_L[p][x] * M_L + delta_L[q][y].
      - accepting is a boolean array of length M_LPP.

    Example
    -------
        delta_cols, delta_table, accept = build_Lpp_dfa(dL, aL)
        delta_table[0][0] == dL[0][0] * M_L + dL[0][0]    # (eps,eps) on (a,a)
        accept[0] is True                                 # eps + 'aa' + eps = 'aa'

    Pre-conditions
    --------------
        - len(delta_L) == M_L and each row has length 4
        - len(accepting_L) == M_L
    """
    assert len(delta_L) == M_L
    assert len(accepting_L) == M_L

    delta_cols = np.asarray(delta_L, dtype=np.int32)         # shape (M_L, 4)
    assert delta_cols.shape == (M_L, 4)

    # Build the full (M_LPP, 16) table using broadcasting instead of a Python
    # double loop (1.87M * 16 entries would be slow in pure Python).
    #
    # Entry at (p*M_L + q, 4*x + y) = delta_L[p][x] * M_L + delta_L[q][y].
    # Equivalently:   delta_table[p, q, x, y] = dp[p, x] * M_L + dq[q, y]
    # reshaped to     (M_LPP, 16).
    dp = delta_cols[:, :, None, None]          # shape (M_L, 4, 1, 1)
    dq = delta_cols[None, None, :, :]          # shape (1, 1, M_L, 4)
    # Compute as int64 to avoid overflow (M_L^2 fits in int32 but the full
    # expression crosses the int32 boundary during multiplication; stay safe).
    big = dp.astype(np.int64) * M_L + dq.astype(np.int64)   # (M_L,4,M_L,4)
    # Rearrange axes from (p, x, q, y) to (p, q, x, y) and flatten.
    big = big.transpose(0, 2, 1, 3).reshape(M_LPP, 16)
    delta_table = big.astype(np.int32)                      # fits (max < 1.87M)

    accepting = _acceptance_mask_Lpp(accepting_L)

    return delta_cols, delta_table, accepting


# =============================================================================
# Section 4 - Counting via the two-array N_j recurrence
# =============================================================================


def count_strings_L(
    delta: Sequence[Sequence[int]],
    accepting: Sequence[bool],
    max_n: int,
) -> List[int]:
    """Compute N_0(0), N_0(1), ..., N_0(max_n) for a 4-symbol DFA using the
    two-array recurrence from the handout.

    Inputs
    ------
    delta      : transition table, delta[j][x] in 0..|Q|-1, alphabet size 4
    accepting  : accepting flags, length |Q|
    max_n      : non-negative integer, highest length to compute

    Output
    ------
    A list ``result`` of length max_n + 1 with ``result[k] = N_0(k)``.

    Example
    -------
        dL, aL = build_L_dfa()
        count_strings_L(dL, aL, 5) == [1, 4, 16, 64, 256, 1024]

    Pre-conditions
    --------------
        - every delta[j][x] is a valid state index
        - every state has exactly 4 transitions
        - max_n >= 0
    """
    N = len(delta)
    # Initial vector: N_j(0) = 1 if j accepts else 0.
    prev: List[int] = [1 if accepting[j] else 0 for j in range(N)]
    result: List[int] = [prev[0]]

    nxt: List[int] = [0] * N
    for _k in range(1, max_n + 1):
        for j in range(N):
            dj = delta[j]
            # N_j(k) = sum_{x in Sigma} N_{delta(j,x)}(k-1)
            nxt[j] = prev[dj[0]] + prev[dj[1]] + prev[dj[2]] + prev[dj[3]]
        # "copy next to prev and repeat" (handout wording).
        prev[:] = nxt
        result.append(prev[0])
    return result


def _lpp_step_literal(
    prev: np.ndarray,
    delta_table: np.ndarray,
) -> np.ndarray:
    """One step of the recurrence using the literal 16-term sum.

    For each state j, computes next[j] = sum_{s=0..15} prev[delta_table[j, s]].
    This is the straightforward evaluation of the recurrence over the full
    materialised (M_LPP, 16) transition table.  Used by the self-test.
    """
    gathered = prev[delta_table]                       # shape (M_LPP, 16)
    return gathered.sum(axis=1)                        # shape (M_LPP,)


def _lpp_step_two_stage(
    prev: np.ndarray,
    delta_cols: np.ndarray,
) -> np.ndarray:
    """One step of the recurrence exploiting the product structure.

    The 16-term sum   sum_{x,y} prev[delta(p,x)*M + delta(q,y)]   factors by
    associativity of addition as

        Q[p, r] = sum_{y}  prev2d[p, delta_L(r, y)]           (mix over y)
        next2d[p, q] = sum_{x}  Q[delta_L(p, x), q]           (mix over x)

    where prev2d has shape (M_L, M_L).  Both stages are 4 vectorised column /
    row adds over object-dtype numpy arrays carrying Python big-ints; this is
    the same arithmetic as the literal 16-term sum, just reassociated.
    """
    prev2d = prev.reshape(M_L, M_L)

    # Stage 1 - for each y, scatter the "columns" indexed by delta_L(:, y)
    # into Q.  Concretely, Q[:, q] = sum_y prev2d[:, delta_L[q, y]].
    Q = prev2d[:, delta_cols[:, 0]].copy()             # y = 0
    for y in range(1, 4):
        Q += prev2d[:, delta_cols[:, y]]

    # Stage 2 - for each x, gather rows indexed by delta_L(:, x) and sum.
    nxt2d = Q[delta_cols[:, 0], :].copy()              # x = 0
    for x in range(1, 4):
        nxt2d += Q[delta_cols[:, x], :]

    return nxt2d.reshape(M_LPP)


def _assert_two_stage_matches_literal(
    delta_cols: np.ndarray,
    delta_table: np.ndarray,
    accepting: np.ndarray,
    sample_size: int = 20_000,
    iterations: int = 3,
) -> None:
    """Startup self-test: run both the literal and two-stage step a few times
    starting from the real L'' initial vector, and assert they agree on a
    random sample of states.

    Keeping the comparison to a random subset keeps the self-test fast while
    still exercising the real DFA.  The two methods compute the same
    mathematical sum, so agreement on 20k random states is overwhelming
    evidence of equivalence.
    """
    rng = np.random.default_rng(seed=454)
    sample = rng.choice(M_LPP, size=sample_size, replace=False)

    prev_lit = np.empty(M_LPP, dtype=object)
    prev_lit[:] = accepting.astype(int)
    prev_two = prev_lit.copy()

    for k in range(1, iterations + 1):
        prev_lit = _lpp_step_literal(prev_lit, delta_table)
        prev_two = _lpp_step_two_stage(prev_two, delta_cols)
        # Compare on the sample.
        if not np.array_equal(prev_lit[sample], prev_two[sample]):
            diffs = np.where(prev_lit[sample] != prev_two[sample])[0]
            raise AssertionError(
                "two-stage step disagrees with literal step at iteration %d "
                "(first mismatching sample index %d; values %r vs %r)"
                % (k, int(diffs[0]),
                   prev_lit[sample[diffs[0]]], prev_two[sample[diffs[0]]])
            )
        # Also confirm N_0(k) matches between the two.
        if int(prev_lit[0]) != int(prev_two[0]):
            raise AssertionError(
                "N_0(%d) disagrees: literal=%d two-stage=%d"
                % (k, int(prev_lit[0]), int(prev_two[0]))
            )


# =============================================================================
# Section 5 - Lazy incremental counters with caches
# =============================================================================


class _LCounter:
    """Incremental two-array counter for L (pure Python)."""

    def __init__(self) -> None:
        self.delta, self.accepting = build_L_dfa()
        N = len(self.delta)
        # prev holds N_j(current_k) for all j.
        self.prev: List[int] = [1 if self.accepting[j] else 0 for j in range(N)]
        self.current_k: int = 0
        self.cache: dict = {0: self.prev[0]}

    def _step(self) -> None:
        N = len(self.delta)
        nxt: List[int] = [0] * N
        for j in range(N):
            dj = self.delta[j]
            nxt[j] = self.prev[dj[0]] + self.prev[dj[1]] + self.prev[dj[2]] + self.prev[dj[3]]
        # Copy next back to prev (follows the handout wording).
        self.prev[:] = nxt
        self.current_k += 1
        self.cache[self.current_k] = self.prev[0]

    def get(self, n: int) -> int:
        """Return N_0(n) for the L DFA (number of length-n strings in L)."""
        while self.current_k < n:
            self._step()
        return self.cache[n]


class _LppCounter:
    """Incremental two-array counter for L'' (numpy object dtype, two-stage)."""

    def __init__(self) -> None:
        delta_L, accepting_L = build_L_dfa()
        print("[init] building L'' product DFA (%d states, 16 symbols) ..."
              % M_LPP, file=sys.stderr, flush=True)
        t0 = time.time()
        self.delta_cols, self.delta_table, self.accept_arr = build_Lpp_dfa(
            delta_L, accepting_L
        )
        print("[init] L'' built in %.2fs; running two-stage self-test ..."
              % (time.time() - t0), file=sys.stderr, flush=True)
        t0 = time.time()
        _assert_two_stage_matches_literal(
            self.delta_cols, self.delta_table, self.accept_arr
        )
        print("[init] self-test passed in %.2fs." % (time.time() - t0),
              file=sys.stderr, flush=True)

        self.prev = np.empty(M_LPP, dtype=object)
        self.prev[:] = self.accept_arr.astype(int)
        self.current_k: int = 0
        self.cache: dict = {0: int(self.prev[0])}

    def _step(self) -> None:
        self.prev = _lpp_step_two_stage(self.prev, self.delta_cols)
        self.current_k += 1
        self.cache[self.current_k] = int(self.prev[0])

    def get(self, k: int) -> int:
        """Return N''_0(k) (count of length-k strings accepted by L'')."""
        if k in self.cache:
            return self.cache[k]
        start = self.current_k
        target = k
        t0 = time.time()
        while self.current_k < target:
            step_t0 = time.time()
            self._step()
            step_dt = time.time() - step_t0
            elapsed = time.time() - t0
            done = self.current_k - start
            remaining = target - self.current_k
            eta = (elapsed / done) * remaining if done else 0.0
            print(
                "[L''] k=%d/%d  step=%.2fs  elapsed=%.1fs  eta=%.1fs"
                % (self.current_k, target, step_dt, elapsed, eta),
                file=sys.stderr, flush=True,
            )
        return self.cache[k]


# =============================================================================
# Section 6 - Interactive menu
# =============================================================================


_BANNER = (
    "\n"
    "===========================================================\n"
    " CS 454 - Project 1 - Counting strings via DFA\n"
    " Martinez & Orozco\n"
    "===========================================================\n"
)

_MENU = (
    "  (1) Count strings of length n in L  (every 6-window hits a,b,c,d)\n"
    "  (2) Count strings of length n in L' (as (1), plus even n with 'aa' as the middle two)\n"
    "  (3) Quit\n"
)


def _prompt_int(prompt: str, lo: int, hi: int) -> int:
    """Prompt until the user enters an integer in [lo, hi]."""
    while True:
        try:
            raw = input(prompt).strip()
        except EOFError:
            print()                        # newline on Ctrl-D
            return 3                       # treat EOF as quit when choosing
        if not raw:
            continue
        try:
            v = int(raw)
        except ValueError:
            print("  not an integer, try again")
            continue
        if v < lo or v > hi:
            print("  value out of range (%d..%d), try again" % (lo, hi))
            continue
        return v


def main() -> None:
    print(_BANNER)

    # L counter is cheap - build eagerly.
    l_counter = _LCounter()
    # L'' counter is heavy - build lazily on first use of choice 2.
    lpp_counter: _LppCounter | None = None

    while True:
        print(_MENU)
        choice = _prompt_int("Enter choice (1/2/3): ", 1, 3)
        if choice == 3:
            print("Goodbye.")
            return
        n = _prompt_int("Enter n (1..300): ", 1, 300)

        if choice == 1:
            ans = l_counter.get(n)
            print("  |{w in L : |w| = %d}| = %d" % (n, ans))
        else:
            if n % 2 == 1:
                print("  |{w in L' : |w| = %d}| = 0   "
                      "(L' is empty for odd lengths)" % n)
                continue
            m = n // 2
            k = m - 1                             # length in L''
            if lpp_counter is None:
                lpp_counter = _LppCounter()
            ans = lpp_counter.get(k)
            print("  |{w in L' : |w| = %d}| = N''_0(%d) = %d" % (n, k, ans))


if __name__ == '__main__':
    main()
