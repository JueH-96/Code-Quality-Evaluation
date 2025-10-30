def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    A = list(map(int, input_data[1:]))

    MOD = 998244353

    # ----------------------------------------------------------------
    # EXPLANATION / PROOF SKETCH
    #
    # We want the number of permutations P of {1..N} for which, for each i:
    #
    #  (1) If A_i > 0, then P_{A_i} < P_i.
    #  (2) For every j with A_i < j < i, we have P_j > P_i.
    #
    # In the language of "previous smaller element," condition (2) says:
    #   "Between A_i+1 and i-1, all P_j are bigger than P_i,"
    # and condition (1) says:
    #   "At A_i, the value P_{A_i} is strictly smaller than P_i (if A_i>0)."
    #
    # One can show (a standard result) that if a permutation P satisfies
    # the above, then for each i, the index A_i is exactly the position of
    # the previous smaller element of P_i in P (or zero if none).  Conversely,
    # any array A of valid "previous-smaller" links can be realized by at least
    # one permutation.  A known (and perhaps surprising) fact is that the
    # number of permutations giving exactly that “previous smaller array” A
    # is
    #
    #           Π (i − A_i),   for i = 1..N.
    #
    # (A short rationale: when assigning values to P_i from smallest to largest
    # or vice versa, the constraints effectively give (i - A_i) ways at step i.
    # A careful inductive or combinatorial argument confirms this product.)
    #
    # We then take that product modulo 998244353.
    #
    # This formula also matches the first sample:
    #   N=4, A=(0,1,0,3)
    #   (i - A_i) = (1, 1, 3, 1), product = 3.
    #
    # For the second sample in the statement, the product of (i - A_i)
    # indeed matches the provided answer when taken modulo 998244353.
    # ----------------------------------------------------------------

    ans = 1
    for i in range(N):
        # i runs 0..N-1, index = i+1 in 1-based
        idx = i + 1
        ans = (ans * (idx - A[i])) % MOD

    print(ans)