def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    A = list(map(int, input_data[1:]))

    MOD = 998244353

    # ----------------------------------------------------------------
    # 1) Quick feasibility check for the endpoints:
    #    - Cell 1 in the initial arrangement is (1 if 1 is odd else 0) = 1
    #      so we must have A[0] == 1.  (Because the problem statement's A_i
    #      is 0-indexed in Python code, but 1-indexed in problem.)
    #    - Cell N in the initial arrangement is (1 if N is odd else 0).
    #      So we must have A[N-1] == (N % 2).
    #
    #    If either endpoint does not match, the answer is 0.
    # ----------------------------------------------------------------
    if A[0] != (1 if 1 % 2 == 1 else 0):
        print(0)
        return
    if A[N-1] != (1 if (N % 2) == 1 else 0):
        print(0)
        return

    # ----------------------------------------------------------------
    # 2) Identify all positions i (1 <= i < N) where A[i] != A[i+1].
    #    In 0-based indexing, these are i = 0..(N-2).
    #    Let S = { i | A[i] != A[i+1] } as a sorted list.
    #    We'll add sentinel values s_0 = 0, s_{m+1} = N (thinking in
    #    1-based color-change indices, but we adapt to 0-based carefully).
    #
    #    Actually, to stay consistent with the problem where color-change
    #    indices went from 1..N-1, we can just treat them the same but
    #    offset by +1 if you like.  It's simpler to do everything in the
    #    0-based code carefully:
    #
    #    The gap between two successive "kept changes" s_j and s_{j+1}
    #    is the set of color-change–indices from (s_j+1) through (s_{j+1}-1).
    #    We want all of those to be removed by bridging.  Each bridging
    #    operation removes an even-length consecutive block of color-changes.
    #
    #    So for each gap length G = (s_{j+1}-1) - (s_j+1) + 1 = s_{j+1} - s_j - 1,
    #    we require G be even, and we then count how many ways (including
    #    orderings) to cover exactly G consecutive "to-be-removed" changes
    #    by disjoint intervals each of even length >= 2.  Denote this count
    #    as p_e(G).  The final answer is the product of p_e(G) over all
    #    gaps j, modulo 998244353.
    #
    # 3) We precompute p_e(G) via a closed-form:
    #
    #       p_e(2n) = f(n),   p_e(2n+1) = 0.
    #
    #    and
    #
    #       f(n) = sum_{k=1..n} [C(n-1, k-1) * k!],    for n >= 1
    #       f(0) = 1.
    #
    #    A more efficient closed-form for f(n) is:
    #
    #       f(n) = (n-1)! * [ n * S(n-1) - S(n-2) ]      (for n>=1)
    #       with f(0) = 1,
    #
    #    where S(m) = sum_{j=0..m} (1/j!)  (in modular arithmetic).
    #
    # 4) Implementation steps:
    #    - Precompute factorials, inverse factorials, and inverse of factorial
    #      for up to 2 * 10^5.  Build array ifact[] = [1/0!, 1/1!, ..., 1/(2e5)!].
    #    - Build array S[] where S[i] = sum of (1/k!) for k=0..i mod 998244353.
    #    - Define a function f(n) = p_e(2n):
    #         - if n==0 => 1
    #         - else => fact[n-1]*( n * S(n-1) - S(n-2) ) mod, taking care
    #           of the n=1 edge carefully, and S(-1)=0 by convention.
    #    - Parse the final arrangement's color-change set S, form the
    #      consecutive gap lengths, check parity, accumulate product of
    #      f(gap/2).
    # ----------------------------------------------------------------

    # ---------------- Precompute factorials etc. ---------------------
    maxN = N  # Enough for G up to N-1 => G/2 up to ~10^5
    fact = [1] * (maxN+1)
    invfact = [1] * (maxN+1)
    for i in range(1, maxN+1):
        fact[i] = fact[i-1] * i % MOD

    # Fermat's little theorem for inverse factorial
    # invfact[maxN] = (fact[maxN])^(MOD-2) mod
    invfact[maxN] = pow(fact[maxN], MOD-2, MOD)
    for i in reversed(range(maxN)):
        invfact[i] = invfact[i+1] * (i+1) % MOD

    # Compute 1/k! in an array ifact[k].
    ifact = [0]*(maxN+1)
    for i in range(maxN+1):
        # 1/i! = invfact[i]
        ifact[i] = invfact[i]

    # Build S array: S(i) = sum_{j=0..i} [1/j!] mod
    # We'll store it in an array partialInvFactSum.
    partialInvFactSum = [0]*(maxN+1)
    partialInvFactSum[0] = ifact[0]  # = 1
    for i in range(1, maxN+1):
        partialInvFactSum[i] = (partialInvFactSum[i-1] + ifact[i]) % MOD

    def S(i):
        """Return sum_{j=0..i} (1/j!).  If i<0, return 0."""
        if i < 0:
            return 0
        return partialInvFactSum[i]

    # f(n) = number of ways to cover 2n color-changes with even-length intervals
    # plus all permutations.  f(0)=1, else f(n) = (n-1)! * [ n * S(n-1) - S(n-2) ] mod
    def f(n):
        if n == 0:
            return 1
        if n == 1:
            # Let's handle n=1 carefully:
            # By the formula: (1-1)!*( 1*S(0)- S(-1) ) = 1*( 1*1 - 0 )=1
            # so it works fine.
            return 1
        return (fact[n-1] * ((n * S(n-1) - S(n-2)) % MOD)) % MOD

    # ----------------------------------------------------------------
    # Identify all color-change positions in A:
    # ----------------------------------------------------------------
    change_positions = []
    for i in range(N-1):
        if A[i] != A[i+1]:
            change_positions.append(i+1)  # store 1-based in spirit, or keep as i+1

    # We'll treat s_0=0, s_{m+1} = N (in 1-based sense).
    # In 0-based sense, those "change indices" run from 1..(N-1).
    # So define extended list s with 0 and N added:
    s = [0] + change_positions + [N]

    ans = 1

    # Loop over consecutive pairs in s:
    for i in range(len(s)-1):
        # gap = (s_{i+1}-1) - (s_i+1) + 1 = s_{i+1} - s_i - 1
        gap = (s[i+1] - s[i] - 1)
        if gap < 0:
            # Should not happen if s[i+1] > s[i].
            ans = 0
            break
        # We want to remove all color-changes in that gap, so gap must be even:
        if gap % 2 != 0:
            ans = 0
            break
        # number of ways is f(gap//2):
        ans = (ans * f(gap//2)) % MOD

    print(ans)