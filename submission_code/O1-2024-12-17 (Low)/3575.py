class Solution:
    def maxValue(self, nums: List[int], k: int) -> int:
        """
        We want to pick exactly 2*k elements from nums, split them into two groups of k each,
        and maximize (OR_of_group1) XOR (OR_of_group2).

        A direct combinatorial approach is too large (n can be up to 400). Instead, we use a
        dynamic programming approach that keeps track of how many elements have been placed
        into each group so far (c1, c2) and which (OR1, OR2) pairs are achievable.

        Let dp[c1][c2] be a set of pairs (or1, or2) indicating that it is possible to achieve
        OR1 = or1, OR2 = or2 using exactly c1 elements in group1 and c2 elements in group2
        from some subset of the processed elements.

        For each new number val, we can:
          - skip it (so dp[c1][c2] carries over to dpNew[c1][c2]),
          - put val in group1 (if c1 < k), transitioning dp[c1][c2] -> dpNew[c1+1][c2] with
            a new pair (or1 | val, or2),
          - put val in group2 (if c2 < k), transitioning dp[c1][c2] -> dpNew[c1][c2+1] with
            a new pair (or1, or2 | val).

        In the end, dp[k][k] contains all (OR1, OR2) pairs achievable by exactly k elements
        in group1 and k in group2, from some selection of 2*k (possibly among all n). We
        return the maximum XOR of any such pair.

        Because numbers are less than 2^7 = 128, OR values range in [0..127], so each of
        or1, or2 fits in 7 bits. dp[c1][c2] can therefore hold up to 128*128 possible states
        in the worst case, though in practice collisions may reduce it.

        Complexity note:
          - We have n (<= 400) iterations,
          - For each (c1, c2) up to (k, k), and k <= n/2 <= 200,
          - We may store up to ~128*128 states per dp[c1][c2].
          - This can be large, but with efficient set operations and Python optimizations,
            it can still be made to run. (In lower-level languages it is more comfortable,
            but in Python we rely on collision and set merging behavior.)

        We store (or1, or2) as a single integer (or1 << 7) | or2 to reduce memory/storage overhead.
        """
        from collections import defaultdict

        n = len(nums)
        
        # dp[c1][c2] will be a set of encoded pairs (p) where:
        #   p = (or1 << 7) | or2
        #   or1, or2 in [0..127]
        dp = [ [ set() for _ in range(k+1) ] for _ in range(k+1) ]
        dp[0][0].add(0)  # (or1=0, or2=0) encoded as 0<<7 | 0 = 0

        for val in nums:
            # We'll build the new dp in dpNext
            dpNext = [ [ set() for _ in range(k+1) ] for _ in range(k+1) ]
            # Copy over existing states first to allow "skip" operation
            for c1 in range(k+1):
                for c2 in range(k+1):
                    dpNext[c1][c2].update(dp[c1][c2])

            # Now handle taking val into either group1 or group2
            for c1 in range(k+1):
                for c2 in range(k+1):
                    for pair in dp[c1][c2]:
                        or1 = pair >> 7
                        or2 = pair & 0x7F

                        # If we can add to group1
                        if c1 < k:
                            new_or1 = or1 | val
                            encoded = (new_or1 << 7) | or2
                            dpNext[c1+1][c2].add(encoded)

                        # If we can add to group2
                        if c2 < k:
                            new_or2 = or2 | val
                            encoded = (or1 << 7) | new_or2
                            dpNext[c1][c2+1].add(encoded)

            # Move dpNext back into dp
            dp = dpNext

        # Now dp[k][k] has all pairs of (or1, or2). We want to maximize or1 XOR or2.
        answer = 0
        for pair in dp[k][k]:
            or1 = pair >> 7
            or2 = pair & 0x7F
            answer = max(answer, or1 ^ or2)

        return answer