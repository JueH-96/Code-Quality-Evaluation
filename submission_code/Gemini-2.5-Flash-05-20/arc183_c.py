import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())

    MOD = 998244353

    # L_prime[p] = maximum L_i (0-indexed) among conditions where X_i = p
    # R_prime[p] = minimum R_i (0-indexed) among conditions where X_i = p
    # These define the effective range [L_prime[p], R_prime[p]] where a dominator must exist
    # if P[p] is the maximum in its original condition range(s).
    
    # Initialize L_prime to -1 to ensure max works correctly from 0.
    # Initialize R_prime to N to ensure min works correctly up to N-1.
    L_prime = [-1] * N  
    R_prime = [N] * N   
    
    # is_constrained[p] is True if p appears as any X_i.
    is_constrained = [False] * N

    # Store all conditions to process them later if needed
    conditions = []
    for _ in range(M):
        L, R, X = map(int, sys.stdin.readline().split())
        L -= 1  # 0-indexed
        R -= 1  # 0-indexed
        X -= 1  # 0-indexed
        conditions.append((L, R, X))

        is_constrained[X] = True
        L_prime[X] = max(L_prime[X], L)
        R_prime[X] = min(R_prime[X], R)

    # Pre-check for impossible cases: If for any constrained position `p`,
    # L_prime[p] > R_prime[p], it means the intersection of all relevant [L_i, R_i] for this `p` is empty.
    # This means `P[p]` can never have a dominator within that intersection, so it can never satisfy its condition.
    # Therefore, no valid permutation exists.
    for p in range(N):
        if is_constrained[p] and L_prime[p] > R_prime[p]:
            print(0)
            return

    # dp[j] will store the number of ways to assign values from 1 to current_value (say, `val`),
    # such that `j` specific positions (among those assigned `1...val`) are 'active' constrained positions.
    # An 'active' constrained position `p` means:
    # 1. `P[p]` has been assigned a value `v <= val`.
    # 2. `is_constrained[p]` is True.
    # 3. `P[p]` is currently the maximum in its effective range `[L_prime[p], R_prime[p]]`
    #    (considering only values 1 to `val` and positions filled so far).
    #    This implies `p` still needs a dominator from `{val+1, ..., N}`.

    # `dp[j]` stores the count for permutations where `j` constraints are currently active.
    # Initialize `dp[0]` to 1 (0 values placed, 0 active constraints).
    dp = [0] * (N + 1)
    dp[0] = 1

    # `active_intervals_at_pos[k]` will store a list of `X` positions whose `[L_prime[X], R_prime[X]]`
    # includes `k`. This is for checking which constraints `k` would satisfy.
    # This precomputation is not needed if the specific problem's DP transitions don't use it.
    
    # We iterate `i` from `0` to `N-1`. This means we are placing value `val = i+1`.
    for i in range(N): 
        # `next_dp` array for results after placing `val = i+1`.
        next_dp = [0] * (N + 1)

        # Iterate through possible number of active constraints `j` from previous step (after placing `i` values).
        for j in range(i + 1): 
            if dp[j] == 0:
                continue

            # Option 1: Place `val = i+1` into an empty slot `p` such that `is_constrained[p]` is `False`.
            # This means `p` is not an `X_k` for any condition. So placing `val` here doesn't create a new active constraint.
            # It also doesn't satisfy any existing `j` constraints.
            # Number of such slots available: `N - i - j`.
            # These are the "normal" empty slots which don't directly interact with the `j` active constraints.
            # `j` remains unchanged.
            if N - i - j > 0:
                next_dp[j] = (next_dp[j] + dp[j] * (N - i - j)) % MOD

            # Option 2: Place `val = i+1` into an empty slot `p` such that `is_constrained[p]` is `True`.
            # This means `p` is an `X_k` for some condition(s). Placing `val` here creates a *new* active constraint,
            # as `val` is currently the largest value, making `P_p` the maximum in its range.
            # This `p` does not satisfy any *other* existing `j` constraints directly.
            # The count of active constraints `j` increases by 1.
            # The number of such slots available is `j + 1`. (These are the "special" empty slots that, when filled,
            # become active. Their count is `j+1` derived from the DP definition for this problem type.)
            if j + 1 <= N: # Check if j+1 is a valid index.
                next_dp[j + 1] = (next_dp[j + 1] + dp[j] * (j + 1)) % MOD

            # Option 3: Place `val = i+1` into an empty slot `p` such that `p` satisfies one of the `j` existing active constraints.
            # This `p` must be within `[L'_k, R'_k]` for one specific active constraint `k`.
            # `val` acts as a dominator for `P_k`. The count of active constraints `j` decreases by 1.
            # The number of such slots available is `j`. (These slots are those which can satisfy one of the `j` conditions).
            if j > 0:
                next_dp[j - 1] = (next_dp[j - 1] + dp[j] * j) % MOD
        
        dp = next_dp

    # Final answer: After placing all `N` values, the count of active constraints must be 0.
    # So `dp[0]` contains the total number of valid permutations.
    print(dp[0])

solve()