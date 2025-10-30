from math import comb

class Solution:
    def waysToReachStair(self, k: int) -> int:
        # Explanation:
        # Alice starts at stair 1 (with “jump” = 0). There are two allowed operations:
        #  1) A “down‐move”: from stair i to i–1. (This move cannot be used twice in a row,
        #     and it cannot be used at stair 0.)
        #  2) An “up‐move”: from stair i to i + 2^(jump). Immediately after an up‐move,
        #     jump is increased by 1.
        #
        # Consider that in any complete sequence of moves that eventually leaves Alice on stair k,
        # she will perform some number m ≥ 0 of up‐moves. (Note that if m = 0, then she never
        # does an up–move; this is possible only when k is no greater than 1 because she starts at 1.)
        #
        # The m up–moves add:
        #    2^0 + 2^1 + … + 2^(m–1) = 2^m – 1
        # to her stair number.
        # Suppose (after all moves) she used d down–moves (each subtracting 1). Then her final
        # stair is:
        #    final = (starting stair) + (sum of ups) – (d) = 1 + (2^m – 1) – d = 2^m – d.
        #
        # To finish at stair k we must have:
        #      2^m – d = k   ⟹   d = 2^m – k.
        #
        # However, because down–moves have the extra restrictions, we must “pay attention” to
        # how they can be arranged. In any sequence the up–moves are forced to occur (in order)
        # and they “split” the move–sequence into m+1 “gaps” (one before the first up–move,
        # one between each pair of successive up–moves, and one after the last up–move).
        # Since down–moves cannot be used consecutively these d moves must lie in different
        # gaps. In other words we must have
        #      d ≤ m+1.
        #
        # Summarizing, for a fixed m we “force” that:
        #    • The total effect is k = 2^m – d  with d = 2^m – k  (so necessarily we require 2^m ≥ k)
        #    • To be physically possible (i.e. to avoid consecutive downs) we need d ≤ m+1.
        #
        # And if there are exactly d downs (with d computed by 2^m – k), then the down–moves may
        # be placed into the m+1 gaps in C(m+1, d) ways.
        #
        # Hence (all sequences are “measured” by the total number m of up–moves) the total number
        # of valid sequences is:
        # 
        #       Sum (over m such that 2^m >= k and 2^m – k ≤ m+1)
        #             [ C(m+1, 2^m – k) ]
        #
        # A few examples:
        #
        # Example k = 0:
        #   • m = 0: 2^0 = 1, d = 1 – 0 = 1. Since 1 ≤ (0+1) we get C(1,1) = 1 (i.e. one way: simply “down”).
        #   • m = 1: 2^1 = 2, d = 2 – 0 = 2. Since 2 ≤ 2 we get C(2,2) = 1.
        #   • m = 2: 2^2 = 4, d = 4 – 0 = 4, but 4 > (2+1)=3 so m = 2 yields no valid sequence.
        #   Total ways = 1 + 1 = 2.
        #
        # Example k = 1:
        #   • m = 0: 2^0 = 1, d = 1 – 1 = 0, ways = C(1,0) = 1.
        #   • m = 1: 2^1 = 2, d = 2 – 1 = 1, ways = C(2,1) = 2.
        #   • m = 2: 2^2 = 4, d = 4 – 1 = 3, ways = C(3,3) = 1.
        #   Total ways = 1 + 2 + 1 = 4.
        #
        # Note: For many values of k no m exists satisfying 2^m – k ≤ m+1; in that case the answer is 0.
        #
        # Because k may be as large as 10^9, note that even though m is “unbounded” in principle,
        # once 2^m grows too large compared with k (since m+1 grows only linearly) the inequality
        # fails. In practice only a few small values of m (roughly m ≤ 30) need to be tested.
        
        ways = 0
        m = 0
        while True:
            # 2^m (using bit-shift to compute quickly)
            up_val = 1 << m  # same as 2**m
            
            # If the upward sum is still less than k then d = 2^m - k would be negative.
            if up_val < k:
                m += 1
                continue

            d = up_val - k  # d must equal the number of down moves
            
            # For a valid interleaving (since down‐moves cannot appear consecutively)
            # we must have at most one down per gap, and there are m+1 gaps.
            if d > m + 1:
                # Because 2^m grows very fast, if this fails once it will fail for all larger m.
                break

            # The number of ways to choose in which of the (m+1) slots the downs appear is binom(m+1, d)
            ways += comb(m + 1, d)
            m += 1

        return ways

# For quick testing:
if __name__ == '__main__':
    sol = Solution()
    # Example cases
    print(sol.waysToReachStair(0))  # Expected output: 2
    print(sol.waysToReachStair(1))  # Expected output: 4

    # Additional tests:
    print(sol.waysToReachStair(2))  # Expected output: 4
    print(sol.waysToReachStair(3))  # Expected output: 3
    print(sol.waysToReachStair(4))  # Expected output: 2

    # A sample larger k; note that many k have no solution (ways = 0) because
    # the only possible k are those for which some m satisfies: 2^m - (m+1) ≤ k ≤ 2^m.
    # For example, k = 100 does not lie in any such interval, so:
    print(sol.waysToReachStair(100))  # Expected output: 0