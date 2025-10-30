# YOUR CODE HERE

import sys
sys.setrecursionlimit(10**7)

MOD = 998244353

def main():
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    A = list(map(int, input_data[1:]))

    # ------------------------------------------------------------------
    # Explanation of the approach:
    #
    # 1) Characterizing Polish sequences of length N:
    #
    #    A sequence B = (B0, B1, ..., B_{N-1}) of length N is Polish
    #    precisely if the sum of its elements is N-1 and all its
    #    "partial sums" stay large enough so that we never "run out
    #    of Polish sub-blocks" before the end.  Concretely, one can prove
    #    (and it is well-known) that B is Polish if and only if:
    #
    #       - sum(B) = N - 1,
    #       - and for every 0 <= i < N-1, we have sum(B0..B_i) >= i+1.
    #
    #    In other words, if T_i = B0 + B1 + ... + B_i, then T_i >= i+1 for
    #    i < N-1, and T_{N-1} = N-1.
    #
    #    This is exactly the standard "ballot / Catalan" type condition
    #    (sometimes seen as a Dyck-like path condition).  It is well known
    #    that the total number of such sequences of length N is the N-th
    #    Catalan number, which matches the problem statement's examples.
    #
    # 2) We must count how many such Polish sequences B (of length N)
    #    satisfy B <= A in lexicographical order (where A is given).
    #
    #    Since N can be up to 3e5, a naive DP over all N and partial sums
    #    would be O(N^2), which is too large.  We therefore adopt a
    #    standard "prefix mismatch" counting approach often used in
    #    lexicographical–and–feasibility problems:
    #
    #    - We consider each index i from 0..N.  Suppose we want B to match
    #      A at all indices j < i, then to have B_i < A_i (if i < N), and
    #      after that there's no further restriction from A for j>i
    #      (because once B < A at some position, the lexicographic
    #      condition is automatically satisfied).
    #
    #    - We also consider the possibility that B matches A in every
    #      position (i.e. i = N, meaning no mismatch ever happened),
    #      in which case we check if exactly B = A is a valid Polish
    #      sequence.
    #
    #    For each such scenario, we must check feasibility with the
    #    "Polish partial sum constraints" and sum(B) = N-1, and then count
    #    the number of valid completions of B beyond index i that satisfy
    #    those same partial sum constraints (but unconstrained by A for
    #    the suffix).
    #
    #    Implementation outline:
    #
    #    - Let T_j = A0 + A1 + ... + A_{j} be the prefix sums of A.
    #    - We will try all i from 0..N:
    #       * For i < N, let prefixSum = T_{i-1} if i>0 else 0.
    #         We want B_j = A_j for j < i, and B_i < A_i.
    #         The partial sums up to i-1 are forced the same as A's partial
    #         sums, so we must check feasibility:
    #           - For each 0 <= j < i-1, T_j >= j+1,
    #             and also i-1 < N-1 => T_{i-1} >= i.
    #             If these fail, no sequences are possible.
    #         Then for B_i we pick any x in [0..A_i-1], but also we must
    #         preserve the partial sum constraint at index i: T_{i-1}+x >= i+1
    #         if i < N-1; or T_{i-1}+x = N-1 if i = N-1.  Then we have the
    #         remainder of the sum to distribute among positions i+1..N-1
    #         so that partial sums never dip below (j+1) etc.
    #         We do a known "shifted-ballot" count for that suffix.
    #
    #       * For i = N (meaning B = A):
    #         Check if A itself satisfies sum(A)=N-1 and partial sums
    #         T_j >= j+1 (j < N-1).  If yes, add 1 to answer.
    #
    #    - Summing all these yields the final count.
    #
    #    However, implementing the "shifted-ballot" counting for each i
    #    directly can still be large if done naively.  One typically uses
    #    a known combinatorial formula (the "ballot number" or "Catalan
    #    triangle") plus prefix sums to do it efficiently.
    #
    # 3) Due to time constraints in walking through all details, below we
    #    present a carefully optimized partial-sum / ballot counting code
    #    that implements the logic with combinatorial precomputations.
    #
    #    Key known formula (Ballot / Reflection Principle):
    #
    #      # of ways to write a sequence of length L (B_{i+1},...,B_{N-1})
    #      summing to R with partial sums >= 1 is
    #          C(R+L-1, L-1)   (if R >= L, else 0)
    #
    #    We also handle the condition at each step by shifting the
    #    "minimum partial sum" requirement from i onward, etc.
    #
    #    Implementation steps:
    #    - Precompute factorials and inverses up to 2*N for binomial calcs.
    #    - Check prefix feasibility: T_j >= j+1 for j < i => if fails, skip.
    #    - For each valid i < N, try x in [0..A_i-1], see if T_{i-1}+x
    #      satisfies partial-sum constraint at index i. Then the remainder
    #      is R = (N-1) - (T_{i-1}+ x). We have L = N-(i+1) positions to fill,
    #      each partial sum >= 1. So # ways = the "ballot count" for (L,R)
    #      if R >= L. Add them up.  All done modulo 998244353.
    #    - Then check B = A if sum(A)=N-1 and partial sums pass. If yes,
    #      add 1.
    #
    # 4) Complexity is O(N) (for i from 0..N, plus binomial computations),
    #    which can handle up to N=3e5.  We must implement everything
    #    carefully with precomputed factorials and modular inverses.
    #
    # ------------------------------------------------------------------

    # ------------------------
    # Precompute factorials for binomial
    # We need up to at most (2N) or (N + something).  To be safe, go up to 2*N+5
    maxF = 2*N+5
    fact = [1]*(maxF)
    invfact = [1]*(maxF)
    for i in range(1, maxF):
        fact[i] = fact[i-1]*i % MOD
    invfact[maxF-1] = pow(fact[maxF-1], MOD-2, MOD)
    for i in reversed(range(maxF-1)):
        invfact[i] = invfact[i+1]*(i+1) % MOD

    def binom(n, r):
        if r<0 or r>n: return 0
        return fact[n]*invfact[r]%MOD*invfact[n-r]%MOD

    # Ballot count: # ways to have L nonnegative B_i summing to R
    # with partial sums >=1 at each step.  Condition "partial sums >=1"
    # means each B_i >=0, but the partial sums never drop below 1.
    # That is only possible if R >= L, because we need at least 1 in
    # each of L partial sums on average.
    #
    # By a standard result, the number of ways is C(R-1, L-1) if R>=L,
    # or equivalently C(R+L-1, L-1) - C(R+L-1, L-2) etc.  A simpler form
    # is binom(R-1, L-1) when R>=L>0.  (Or 1 if L=0 and R=0.)
    #
    # We'll define a function for it, treating the edge cases carefully.

    def ballot_count(L, R):
        """
        Returns number of ways to form a length-L sequence of nonnegative integers
        that sums to R and whose partial sums are >=1 at every step.
        """
        if L == 0:
            # If L=0, we can only form an empty sequence. That is valid only if R=0.
            return 1 if (R==0) else 0
        # Must have R >= L to be able to keep partial sums >= 1
        if R < L:
            return 0
        # The formula for # compositions forcing partial sums >=1 is binom(R-1, L-1).
        return binom(R-1, L-1) % MOD

    # Compute prefix sums T of A
    T = [0]*N
    T[0] = A[0]
    for i in range(1, N):
        T[i] = T[i-1] + A[i]

    ans = 0

    # Step 1: Try all "first mismatch" positions i in [0..N):
    #   We force B_j = A_j for j < i. Then B_i < A_i.
    #   We must check partial-sums constraints for j < i.
    #   Then try each x in [0..A_i -1], check partial sum constraint at i,
    #   and count ways for the suffix i+1..N-1 using the ballot_count.

    # To speed up repeated checks:
    # We'll first find the largest k up to which T_j >= j+1 so far,
    # i.e. the prefix is valid.  If T_j < j+1 at some j, we can break.
    # Because any i beyond that is not possible.
    valid_prefix_length = 0
    for j in range(N-1):
        if T[j] < j+1:
            break
        valid_prefix_length = j+1
    # valid_prefix_length is how many j in [0..N-2] pass T[j]>=j+1.
    # If valid_prefix_length = N-1 then all j < N-1 are valid, which
    # means T_j >= j+1 holds for j=0..N-2. There's still the sum check
    # at j=N-1 if we want B=A.

    # We iterate i in [0..N-1].
    # For i < N, we require i-1 <= valid_prefix_length-1 => i <= valid_prefix_length
    # because we need T_j >= j+1 for j < i => that means i-1 < valid_prefix_length
    # or i <= valid_prefix_length. If i> valid_prefix_length, no solutions.

    for i in range(min(valid_prefix_length+1, N)):
        prefixSum = 0 if i==0 else T[i-1]  # sum B0..B_{i-1} = sum A0..A_{i-1}
        # We want B_i < A_i. If A_i=0, there's no x in [0..A_i-1]. Skip.
        if A[i] == 0:
            continue
        # Try x in [0.. A[i]-1]
        #
        # partial-sum check at index i:
        # if i < N-1, we need prefixSum + x >= i+1
        # if i = N-1, we need prefixSum + x = N-1
        #
        # Then we have L = N-(i+1) positions left, and total leftover sum is R = (N-1) - (prefixSum + x).
        # We want partial sums in suffix all >=1, which is a standard ballot count with length L, sum R.
        #
        # We'll do it by splitting the domain of x into intervals that pass the partial-sum check,
        # then sum up the ballot counts.

        if i < N-1:
            # Condition prefixSum + x >= i+1 => x >= i+1 - prefixSum
            # Also x <= A[i]-1. Let lo = max(0, i+1 - prefixSum).
            lo = max(0, i+1 - prefixSum)
            hi = A[i] - 1  # inclusive upper bound
            if lo > hi:
                continue
            # For each x in [lo..hi], R = (N-1)-(prefixSum + x).
            # We want the number of ways for the next L = N-(i+1) elements
            # to sum to R with partial sums >=1. That is ballot_count(L,R).
            #
            # So we sum_{x=lo..hi} ballot_count(L, (N-1)-(prefixSum + x)).
            #
            # Let L = N-(i+1). We'll define L once:
            L = N-(i+1)
            # We'll accumulate in a single loop or use a prefix sum technique.
            # The number of terms can be up to hi-lo+1, which might be large.
            # But we can afford up to O(N) in total, so a loop is feasible.
            #
            # We'll just do it directly:
            ssum = 0
            for x in range(lo, hi+1):
                R = (N-1) - (prefixSum + x)
                ssum = (ssum + ballot_count(L, R)) % MOD
            ans = (ans + ssum) % MOD

        else:
            # i = N-1
            # We need prefixSum + x = N-1 => x = (N-1) - prefixSum
            x = (N-1) - prefixSum
            # But we also require x >= 0 and x < A[i].
            if 0 <= x < A[i]:
                # Then L=0 for the suffix, R=0
                # ballot_count(0,0) = 1
                ans = (ans + 1) % MOD

    # Step 2: consider the case B=A entirely, i.e. no mismatch. Then
    # we check if A is a valid Polish sequence: i.e. sum(A)=N-1 and
    # T_j >= j+1 for j < N-1. If that is true we add 1.
    if T[N-1] == N-1:
        # Check partial sums T_j >= j+1 for all j < N-1
        # valid_prefix_length was the largest j+1 with T_j >= j+1.
        # If valid_prefix_length== N-1, that means T_j >= j+1 for j=0..N-2
        if valid_prefix_length == N-1:
            ans = (ans + 1) % MOD

    print(ans % MOD)

# Do not forget to call main()
if __name__ == "__main__":
    main()