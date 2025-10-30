class Solution:
    def waysToReachStair(self, k: int) -> int:
        # Explanation:
        # Alice starts at stair 1 with a “jump counter” = 0. There are two types of moves:
        #   1. Down move: from stair i to i–1. But you may not use a down move twice in a row
        #      and you cannot use it on stair 0.
        #   2. Up move: from stair i to i + 2^(jump) and then increase jump by 1.
        #
        # Notice that the up moves are “deterministic” in the sense that if you make m of them,
        # they add: 2^0 + 2^1 + … + 2^(m-1) = 2^m – 1. Since Alice begins at stair 1,
        # if she makes m upward moves and d down moves, her final stair will be:
        #       stair = 1 + (2^m – 1) – d = 2^m – d.
        #
        # To finish at stair k we must have: 2^m – d = k, i.e. d = 2^m – k.
        #
        # However, because of the “down move” restrictions we cannot insert arbitrarily many D’s.
        # In a sequence with m up moves the “maximum” number of down moves that can be inserted
        # (without ever using two down moves consecutively) is m + 1. Thus we must have:
        #       0 <= d = 2^m – k <= m + 1.
        #
        # For each m (number of U moves) for which:
        #       2^m >= k  and  2^m – k <= m + 1,
        # we consider sequences that use exactly m up moves and d = 2^m – k down moves.
        #
        # We now count how many valid interleavings (sequences of operations) there are that satisfy:
        #   • They use exactly m U moves and d D moves.
        #   • They never have two consecutive down moves.
        #   • A down move is only allowed if the current stair > 0.
        #
        # Note: The U moves are “ordered” – the first U always adds 2^0, the second 2^1, etc.
        # Thus, if we know how many ups have been used, we know the total upward addition.
        #
        # We simulate the process with a DP function that tracks our state. When the current state is:
        #       (i ups done, d downs used, last move was a down or not),
        # the current stair is computed as (2^i – d).
        #
        # From this state:
        #   - We can always do an up move (if i < m) – this increments i and resets the "down flag".
        #   - We may perform a down move (if d < d_total, the last move was not a down, and the current stair > 0).
        #
        # When we have used all m U moves and exactly d_total down moves, we have a valid sequence.
        #
        # Finally, we sum this over all possible m (which are very limited by the condition 2^m – k <= m+1).
    
        total_ways = 0
        # Try possible m from 0 to a safe upper bound (60 here is more than enough).
        for m in range(0, 61):
            up_total = 1 << m  # 2^m
            if up_total < k:
                continue
            d_total = up_total - k  # required number of down moves
            if d_total < 0 or d_total > m + 1:
                continue
    
            # dp(i, d_used, last_down) returns the number of ways to finish
            # using a total of m up moves and d_total down moves.
            # i: how many up moves have been performed
            # d_used: how many down moves have been performed
            # last_down: 0 if the previous move was not a down, 1 if it was.
            # The current stair = (2^i – d_used)
            from functools import lru_cache
            @lru_cache(maxsize=None)
            def dp(i, d_used, last_down):
                # Compute current stair:
                current = (1 << i) - d_used
                # If we have performed all moves, return 1.
                if i == m and d_used == d_total:
                    return 1
                ways = 0
                # Option: Up move (if available, always allowed regardless of current stair)
                if i < m:
                    ways += dp(i + 1, d_used, 0)
                # Option: Down move (if allowed)
                # Allowed only if: we haven't used all down moves, the last move was not a down,
                # and the current stair is > 0.
                if d_used < d_total and last_down == 0 and current > 0:
                    ways += dp(i, d_used + 1, 1)
                return ways
    
            total_ways += dp(0, 0, 0)
    
        return total_ways

# The following part is a simple I/O wrapper for testing.
if __name__ == '__main__':
    import sys
    data = sys.stdin.read().strip().split()
    if data:
        k = int(data[0])
        sol = Solution()
        sys.stdout.write(str(sol.waysToReachStair(k)))
    else:
        # Sample test cases:
        sol = Solution()
        print("For k = 0, ways =", sol.waysToReachStair(0))  # Expected output: 2
        print("For k = 1, ways =", sol.waysToReachStair(1))  # Expected output: 4