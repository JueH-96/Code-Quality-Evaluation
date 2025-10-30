# YOUR CODE HERE
import sys

def solve():
    """
    This function solves the permutation counting problem using dynamic programming with bitmasking.
    """
    
    # Read input from stdin
    try:
        input_stream = sys.stdin
        N_str, M_str = input_stream.readline().split()
        N, M = int(N_str), int(M_str)
        conditions = [tuple(map(int, line.split())) for line in input_stream.readlines()]
    except (IOError, ValueError):
        # Fallback for empty input or format errors
        print(0)
        return

    MOD = 998244353
    
    # Group conditions by X_i for efficient lookup
    conditions_by_x = [[] for _ in range(N + 1)]
    for L, R, X in conditions:
        conditions_by_x[X].append((L, R))

    # Precompute bitmasks for all unique ranges [L, R]
    # to speed up intersection checks.
    range_masks = {}
    for X in range(1, N + 1):
        for L, R in conditions_by_x[X]:
            if (L, R) not in range_masks:
                mask = 0
                for i in range(L, R + 1):
                    mask |= (1 << (i - 1))
                range_masks[(L, R)] = mask

    # dp[mask] stores the number of valid ways to place the popcount(mask) largest
    # numbers into the positions represented by the set bits in `mask`.
    dp = [0] * (1 << N)
    dp[0] = 1
    
    # Iterate through masks based on the number of elements placed (popcount)
    for i in range(N):
        for mask in range(1 << N):
            # Process masks corresponding to placing `i` elements
            if bin(mask).count('1') != i:
                continue
            
            if dp[mask] == 0:
                continue
            
            # Try to place the (i+1)-th largest number (N-i) in an empty position `p`
            for p_idx in range(N):
                # Check if position `p_idx` is not yet in the mask
                if not ((mask >> p_idx) & 1):
                    # Position `p` is 1-indexed, `p_idx` is 0-indexed
                    pos = p_idx + 1
                    
                    # Check if placing value (N-i) at `pos` is valid.
                    # The `i` largest numbers {N, ..., N-i+1} are already at positions in `mask`.
                    # For `pos` to be a valid placement for `N-i`, for every condition
                    # (L,R,X) with X=pos, the range [L,R] must contain a position
                    # where a number larger than N-i is placed. These are positions in `mask`.
                    is_valid = True
                    for L, R in conditions_by_x[pos]:
                        # Check intersection of `mask` with the range `[L, R]`
                        if (range_masks[(L, R)] & mask) == 0:
                            is_valid = False
                            break
                    
                    if is_valid:
                        new_mask = mask | (1 << p_idx)
                        dp[new_mask] = (dp[new_mask] + dp[mask]) % MOD

    # The final answer is for the mask with all N bits set
    print(dp[(1 << N) - 1])

solve()