MOD = 998244353

def solve():
    N = int(input())
    A = list(map(int, input().split()))

    # dp[i] stores the number of ways to achieve A[0...i].
    # This means cells 0 through i have been transformed to A[0] through A[i].
    dp = [0] * N

    # S[val][parity] stores the sum of dp[j] for all j < current_i,
    # where A[j] == val and j % 2 == parity.
    # This sum includes contributions from dp[-1] for the base case (before cell 0).
    # Conceptual cell -1: value is irrelevant, but its parity is 0.
    # We add 1 to S[A[0]][0] to include dp[-1] (which is 1 way for empty prefix).
    # So S[A_val][index_parity] stores sum(dp[j]) for j which are *actual indices* (0 to N-1).
    # The `dp[-1]` contribution is effectively for `j = -1`.
    # Let's say `dp[-1]` adds to `S[A[0]][0]`. This is usually how it's handled.

    # S[A_value][index_parity]
    S = [[0, 0] for _ in range(2)]

    # Initialize S for the base case `dp[-1] = 1`.
    # This `dp[-1]` can be a left endpoint for an operation starting at cell 0.
    # The operation is `(0, r)`. Cell 0 has value `A[0]`, parity `0`.
    # So `dp[-1]` contributes to `S[A[0]][0]`.
    if N > 0:
        S[A[0]][0] = (S[A[0]][0] + 1) % MOD # dp[-1] = 1. Used for cell 0 as left endpoint.

    for i in range(N): # i is current cell index (0-indexed)
        # Calculate dp[i] (ways to fix A[0...i]).
        current_A_val = A[i]
        current_init_parity = i % 2
        
        # Option 1: Cell `i` matches its initial value (`i%2`) and is not the right endpoint of an operation (l,i).
        # This means A[i] must be equal to i%2.
        # Inherit ways from dp[i-1].
        if current_A_val == current_init_parity:
            if i == 0: # dp[-1] is 1
                dp[i] = (dp[i] + 1) % MOD
            else:
                dp[i] = (dp[i] + dp[i-1]) % MOD
        
        # Option 2: Cell `i` is the right endpoint `r` of an operation `(l,i)`.
        # This operation connects `l` (index `l_idx`) to `i`.
        # `A[l_idx]` must be equal to `A[i]`.
        # `l_idx` and `i` must have the same parity.
        # The contribution comes from `S[A[i]][i%2]`. This sum includes `dp[l_idx]` for valid `l_idx`.
        dp[i] = (dp[i] + S[current_A_val][current_init_parity]) % MOD

        # Update S for future calculations (cell `i` now becomes a potential `l` endpoint).
        # Add `dp[i]` to `S[A[i]][i%2]`.
        S[current_A_val][current_init_parity] = (S[current_A_val][current_init_parity] + dp[i]) % MOD

    # The final answer is dp[N-1] (ways to fix A[0...N-1]).
    print(dp[N-1])