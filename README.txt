CS 454 - Theory of Computation - Project 1
Fall 2026 - due Oct 15, 2026
Team: B. Axel Martinez and Emilio Orozco

===========================================================================
WHAT THIS PROGRAM DOES
===========================================================================

Given the alphabet Sigma = {a, b, c, d}, the project asks for the number of
strings of length n for two languages:

  L   = { w in Sigma* : every length-6 substring of w contains each of
                        a, b, c, d at least once }
        All strings of length < 6 are in L by vacuous truth.

  L'  = { w in L : |w| is even AND the two middle symbols of w are 'aa' }

The program builds the appropriate DFAs explicitly, and for an input n in
1..300 computes the exact (unlimited-precision) number of length-n strings
accepted.

===========================================================================
FILES
===========================================================================

  project1.py   - the graded program.  Interactive menu with the three
                  choices required by the handout (1 = count in L,
                  2 = count in L', 3 = quit).  Loops until 3.
  verify.py     - independent correctness harness (NOT part of the graded
                  program; it is provided so the grader can convince
                  themselves the n = 300 answers are right).
  run.sh        - thin launcher.  "./run.sh" starts the menu;
                  "./run.sh -t" runs verify.py.
  sample_io.txt - captured transcript of a session including n = 300 for
                  both choices.
  README.txt    - this file.

===========================================================================
HOW TO RUN
===========================================================================

    ./run.sh              # interactive menu
    ./run.sh -t           # run the full verification harness

or equivalently:

    python3 project1.py
    python3 verify.py

Python 3.8+ with numpy is required.  No other dependencies.

===========================================================================
STATE COUNTS AND LABELLING (as the handout requires)
===========================================================================

L DFA
-----
  States labelled 0..1365.  State 0 is the start state (empty buffer).
  States 1..1364 encode non-empty buffers of length 1..5 via base-4:

      code(epsilon) = 0
      code(b_1 b_2 ... b_l) = sum_{i=1..l}  digit(b_i) * 4^(l-i)
      digit(a)=1  digit(b)=2  digit(c)=3  digit(d)=4

  so the handout's example "babca" maps to 2*4^4 + 1*4^3 + 2*4^2 + 3*4 + 1
  = 621.  State 1365 is the fail (sink) state.  All non-fail states are
  accepting.  Alphabet labels: a=0, b=1, c=2, d=3.  Total: 1366 states.

L'' DFA
-------
  The handout defines L'' such that |{w in L' : |w| = 2m}| = |{z in L'' : |z| = m-1}|.
  Our construction:

    A word in L' of length 2m is   w = u aa v   with |u| = |v| = m-1.
    Reading u left to right and v right to left in lockstep gives a word
    of length m-1 over the 16-symbol pair alphabet Sigma x Sigma.
    The product state is a pair (p, q) where p is the forward L-DFA
    state on u and q is the forward L-DFA state on reverse(v).  (The
    L-DFA transitions are reversal-symmetric so this backward track
    uses the same delta_L.)  The transition on symbol (x, y) is

        delta''((p,q), (x,y)) = (delta_L(p, x), delta_L(q, y))

    Acceptance of (p, q): neither component is the fail state AND the
    joined string buf(p) + "aa" + reverse(buf(q)) has every length-6
    window hitting a, b, c, d.  (This catches every junction-crossing
    window; windows fully inside u or v were already policed by the
    component DFAs.)

  State labelling:  (p, q) is labelled  p * 1366 + q, giving labels
  0 .. 1,865,955.  Input symbols: (x, y) labelled 4*x + y in 0..15.
  Total: 1,866,956 states on an alphabet of size 16 (matches the
  handout's "around 1400^2" states and 16-symbol alphabet).

  Answer:  |{w in L' : |w| = n}|  =  N''_0(n/2 - 1)  for even n,  else 0.

===========================================================================
PSEUDO-CODE OF THE CORE ALGORITHMS
===========================================================================

Shared counting routine (follows the handout exactly):

    function count(delta, accepting, n):
        for j in 0..|Q|-1:
            prev[j] = 1 if j in F else 0
        for k in 1..n:
            for j in 0..|Q|-1:
                next[j] = sum over symbols x of prev[delta(j, x)]
            prev := next                   # "copy next to prev and repeat"
        return prev[0]                     # = N_0(n)

For L this runs over 1366 states with alphabet size 4; the full sweep to
n = 300 completes in well under a second.

For L'' we use the same recurrence over 1,866,956 states, with the
16-term inner sum factored by associativity into two 4-term stages (the
mathematically identical reorganisation
    sum_{x,y} prev[delta_L(p,x)*M + delta_L(q,y)]
      = sum_x [ sum_y prev[delta_L(p,x), delta_L(q,y)] ] )
so each iteration performs 4 column-adds and 4 row-adds on object-dtype
numpy arrays carrying Python big-ints.  A startup self-test verifies that
this reorganisation produces the same N_j vector as the literal 16-term
sum computed from the materialised (1865956, 16) transition table.

===========================================================================
PERFORMANCE AND MEMORY NOTES
===========================================================================

L alone: effectively instantaneous for any n up to 300.

L'': the DFA is built lazily on the first choice-2 query.
  - DFA construction:   ~3-4 s
  - Self-test:          ~3 s
  - One iteration:      ~0.2 s early, growing with big-int size
  - Full sweep to n=300 (k=149):   approximately 1-4 minutes depending
                                   on the machine

Peak memory for the L'' sweep is roughly 600 MB (two object arrays of
1.87M big-ints plus one intermediate).  The explicit int32 transition
table is ~120 MB.  Both easily fit on the department server.

Queries are cached, so once the sweep reaches some k = K, every query
for n <= 2K+2 is answered instantly thereafter.

===========================================================================
TEAM CONTRIBUTIONS
===========================================================================

We worked on this project together as a pair-programming effort: we sat
side by side for every file and talked through the design at every step,
so there is no clean "Axel wrote X, Emilio wrote Y" split.

Here is roughly how we went about it:

  - We worked out the DFA for L together on the whiteboard before either
    of us touched the keyboard - the base-4 encoding scheme from the
    handout, our 1365 buffer states plus the fail state, and the
    window check we run on reading each new symbol.

  - We derived the product construction for L'' together as well: our
    (p, q) state labelling, the u-aa-v decomposition, and the argument
    for why our junction check only needs the joined length-<=12 string.
    One of us drove the keyboard while the other read the handout aloud
    and double-checked our indices.

  - The two-stage factored iteration for L'' (splitting the 16-term sum
    into two 4-term stages so our big numpy array stays manageable) was
    the trickiest piece for us.  We prototyped it in a scratch file,
    convinced ourselves on paper that our reassociation is identical to
    the literal 16-term sum, and then wrote the self-test (we verify
    that both produce the same N_j vector on a random sample of 20k
    states) so that we would catch any indexing mistake up front.

  - We wrote verify.py as a team sanity check once our main program
    produced its first n = 300 answer, so that we could show the grader
    three independent methods (brute force for small n, forward DP for
    medium n, and our factored path-count identity for n = 300) all
    agreeing on the same number.

  - We put together this README, the sample_io transcript, and our
    written report jointly after the code was finalised.

Any bug, any typo, and any design choice in this submission belongs to
both of us.
