def main():
    import sys
    sys.setrecursionlimit(10**7)
    MOD = 998244353

    input_data = sys.stdin.read().strip().split()
    N, K = map(int, input_data[:2])
    S = input_data[2]

    # ----------------------------------------------------------------
    # PROBLEM RECAP:
    #
    # We have a string S of length N (1 ≤ N ≤ 50) drawn from {A,B,C,?}.
    # We can replace each '?' with one of {A,B,C} in any way we like.
    # A substring T is called "good" if it can be reduced to the empty
    # string by repeatedly deleting either:
    #   (i) pairs of identical letters ("XX" -> ""), or
    #   (ii) triples of distinct letters one A one B one C ("ABC" -> "").
    #
    # It is known (and can be shown) that a string T of A,B,C is "good"
    # precisely if the counts of A,B,C in T all share the same parity.
    # In other words, #A % 2 = #B % 2 = #C % 2.
    #
    # We want to count the number of ways to replace '?' with A,B,C so
    # that the resulting string contains at least K "good" substrings,
    # counting substrings by position (so even two identical substrings
    # in different positions count separately).
    #
    # We must output this count modulo 998244353.
    #
    # ----------------------------------------------------------------
    # KEY OBSERVATION:
    #
    # A substring i..j is "good" iff #A, #B, #C in that substring have
    # the same parity.  Equivalently, in terms of prefix-parities:
    #
    #   Define par(i) = ( (#A up to i) mod 2,
    #                     (#B up to i) mod 2,
    #                     (#C up to i) mod 2 ).
    #
    # Then substring i+1..j is good  ⇔  par(i) = par(j).
    #
    # Consequently, the total number of good substrings in a string
    # is the number of pairs (i, j) with i < j such that par(i)=par(j).
    # If we let count_p be the number of indices i (0 ≤ i ≤ N) such that
    # par(i)=p (where p is one of 8 possible parity triples in {0,1}^3),
    # then the number of good substrings is
    #
    #       sum_{p}  ( count_p choose 2 )
    #     = sum_{p}  count_p*(count_p - 1)/2.
    #
    # Because par(0) = (0,0,0) is always counted, we have sum_p count_p = N+1.
    #
    # Hence if we define c_p = count of parity p among prefix parities,
    # the number of good substrings is:
    #
    #    G = sum_p [ c_p*(c_p-1)/2 ].
    #
    # That may also be written as:
    #
    #    G = (1/2) * ( sum_p c_p^2 - (N+1) ).
    #
    # Thus G ≥ K  ⇔  sum_p c_p^2 ≥  2K + (N+1).
    #
    # ----------------------------------------------------------------
    # THE CRUX: COUNTING PATHS WITH A GIVEN PARITY-DISTRIBUTION
    #
    # As we scan from left to right (prefixes of length i=1..N),
    # par(i) moves from par(i-1) to par(i-1) XOR e(letter), where e(A)
    # flips the A-bit, e(B) flips the B-bit, e(C) flips the C-bit.
    # If S[i-1]=='?', we can choose among A,B,C.  Otherwise we have a
    # forced single choice.
    #
    # We want to track how many ways to form all parity-sequences
    # p_0=0, p_1, p_2, ..., p_N with p_{i+1}=p_i XOR e( chosen_letter_i )
    # consistent with S[i], and then how many times each parity p occurs
    # in {p_0,...,p_N}.
    #
    # Finally, we only want to count those sequences for which
    # sum_p c_p^2 >= (2K + N+1).
    #
    # A direct dynamic programming that keeps track of the full histogram
    # of parities visited is typically very large (state explosion),
    # because N=50 and we might have up to C(N+8,8) possible histograms.
    #
    # Nonetheless, with N=50 (quite small) and careful memo + pruning
    # (and efficient implementation in a lower-level language), one
    # can manage a solution.  In Python this is borderline.  However,
    # the problem as stated (being a harder competitive programming
    # puzzle) usually does suggest a DP over "counts of each parity" plus
    # "current parity."
    #
    # We present here a reference implementation of that canonical
    # approach.  It uses a dictionary-based DP, storing polynomials
    # (or counts) indexed by (i, current_parity, c0, c1, ..., c7).
    # Then from each state we branch to up to 3 next states if S[i] = '?'.
    # The final result enumerates all possible distributions c_p.  Then we
    # sum up those whose sum_p c_p^2 >= 2K + (N+1).  Because N=50 won't
    # exceed a few million states with good pruning, this can pass in C++.
    # In Python, it is quite tight but can still be done with careful
    # dictionaries and ordering.  We'll implement it as cleanly as possible.
    #
    # ----------------------------------------------------------------
    # IMPLEMENTATION NOTES / OPTIMIZATIONS:
    #
    # 1) Instead of storing a full (c0,...,c7), we store only the *difference*
    #    from the starting distribution.  But we do start from (c0=1, c1=...=0),
    #    because par(0)=0 is visited once.
    # 2) Each step i moves from (old_curpar, c0..c7) to (new_curpar, updated_c0..c7).
    # 3) Because sum(c_p)= i+1, the possible sets c_p are limited to those with
    #    sum(c_p)=i+1.  That is a combinatorial number but can be iterated in
    #    practice up to i=50 with some care (especially in lower-level languages).
    # 4) We store the dp states in a dictionary from a compressed representation:
    #    We keep "dp_next[(new_parity, new_counts_as_tuple)] += dp_val".
    # 5) At the end, we look at all dp[N,*] states, compute sum_p c_p^2, and if
    #    that is ≥ 2K + (N+1), we add dp_val to the answer.
    # 6) All arithmetic is mod 998244353.
    #
    # Although this is large, for N=50, it is the classic editorial approach.
    # In a real contest environment, one would likely use C++/Rust.  In Python,
    # this is borderline feasible with aggressive optimization (e.g. using
    # dictionaries of manageable size).  We will implement a careful version.
    #
    # WARNING: This solution is not trivial.  It is somewhat heavy and might
    # be near the time/memory limits in Python.  We include it here as it is
    # the standard correct method for this puzzle.
    #
    # ----------------------------------------------------------------

    # STEP 1: Map each parity-triple (pA,pB,pC) in {0,1}^3 to an integer 0..7.
    #         Also define how to flip parity for 'A','B','C'.
    # p is in [0..7], bits: pA = (p>>0)&1, pB = (p>>1)&1, pC = (p>>2)&1
    # flipping 'A' means toggling bit0, 'B' -> bit1, 'C'-> bit2

    # Precompute next-parity from p for letter 'A','B','C'.
    # next_p_A = p ^ 1, next_p_B = p ^ 2, next_p_C = p ^ 4
    # We will store them in arrays for fast access.

    nextA = [p ^ 1 for p in range(8)]
    nextB = [p ^ 2 for p in range(8)]
    nextC = [p ^ 4 for p in range(8)]

    # Convert S[i] to a mask or list of possible transitions:
    # If S[i]=='A' -> only nextA
    # If S[i]=='B' -> only nextB
    # If S[i]=='C' -> only nextC
    # If S[i]=='?' -> nextA, nextB, nextC
    # We'll store for each i a small list of "which of {nextA, nextB, nextC} to use"
    # so that we can loop over them in the DP transition.

    possible_transitions = []
    for ch in S:
        if ch == 'A':
            possible_transitions.append([0])  # denotes "use nextA"
        elif ch == 'B':
            possible_transitions.append([1])  # denotes "use nextB"
        elif ch == 'C':
            possible_transitions.append([2])  # denotes "use nextC"
        else:  # '?'
            possible_transitions.append([0,1,2])  # A,B,C all possible

    # We will store DP as a dictionary for each i:
    #   dp[i] = dict of { (parity, c0,c1,c2,c3,c4,c5,c6,c7) : ways }
    # subject to sum(c_k)= i+1, c_0 >= 1 if i>=0, because index 0 uses parity0 once.
    #
    # Start with i=0 => we have 1 prefix state (p_0=0). So c_0=1, others=0.
    # We then move to i=1 by applying the letter at S[0].
    #
    # To reduce memory usage, we keep dp[i] only for the current i and
    # build dp[i+1] from dp[i], then discard dp[i].
    #
    # At the end, we sum over dp[N] states that satisfy sum_p c_p^2 >= 2K + (N+1).

    from collections import defaultdict

    # dp[i] will be a dictionary:
    # key = (curpar, c0,c1,c2,c3,c4,c5,c6,c7)
    # value = # of ways mod 998244353
    dp_current = defaultdict(int)
    # i=0 => visited parity 0 exactly once
    init_key = (0,1,0,0,0,0,0,0,0)  # curpar=0, c0=1, c1=..=c7=0
    dp_current[init_key] = 1

    for i in range(N):
        dp_next = defaultdict(int)
        transitions = possible_transitions[i]  # small list of up to 3 items
        for (curpar, c0,c1,c2,c3,c4,c5,c6,c7), ways in dp_current.items():
            if not ways:
                continue
            # The counts array as a tuple for convenience
            counts = [c0,c1,c2,c3,c4,c5,c6,c7]
            for t in transitions:
                # t=0 => use nextA, t=1 => nextB, t=2 => nextC
                if t==0:
                    newpar = nextA[curpar]
                elif t==1:
                    newpar = nextB[curpar]
                else:
                    newpar = nextC[curpar]
                # we increment the count of newpar by 1
                new_counts = counts[:]
                new_counts[newpar] += 1
                # build new key
                new_key = (newpar, *new_counts)
                dp_next[new_key] = (dp_next[new_key] + ways) % MOD

        dp_current = dp_next  # move to next

    # Now dp_current = dp[N], dictionary of final states
    # We want to sum all ways for states whose sum_{p} c_p^2 >= 2K + (N+1).
    # sum_{p} c_p^2 = c0^2 + c1^2 + ... + c7^2
    # And we also have sum_{p} c_p = N+1 for sure.

    M = 2*K + (N+1)

    ans = 0
    for (curpar, c0,c1,c2,c3,c4,c5,c6,c7), ways in dp_current.items():
        ssum = c0*c0 + c1*c1 + c2*c2 + c3*c3 + c4*c4 + c5*c5 + c6*c6 + c7*c7
        if ssum >= M:
            ans = (ans + ways) % MOD

    print(ans % MOD)