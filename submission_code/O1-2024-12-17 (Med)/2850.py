class Solution:
    def longestString(self, x: int, y: int, z: int) -> int:
        """
        We have:
          - x blocks "AA"
          - y blocks "BB"
          - z blocks "AB"

        We want to form the longest possible concatenation of some (or all) of these blocks
        without ever creating "AAA" or "BBB" as a substring.

        Observing how "AAA" or "BBB" can form:
          - "AA"+"AA" -> "AAAA" (contains "AAA"), not allowed
          - "AA"+"AB" -> "AAAB" (contains "AAA"), not allowed
          - "AB"+"BB" -> "ABBB" (contains "BBB"), not allowed
          - "BB"+"BB" -> "BBBB" (contains "BBB"), not allowed

        All other concatenations are safe. We denote:
          A := "AA"
          B := "BB"
          C := "AB"

        We define valid adjacency in a small directed graph as follows:
          A -> B (allowed)
          B -> A (allowed)
          B -> C (allowed)
          C -> A (allowed)
          C -> C (allowed)
        and all other transitions are disallowed.

        We then use a DP (memoized DFS) approach where:
          dp(a, b, c, last) = maximum number of blocks we can form
          given a remaining "AA", b remaining "BB", c remaining "AB",
          and 'last' is the type of the block we placed previously (A, B, C, or 'start').

        The final result is dp(x, y, z, 'start') * 2, because each block is length 2.
        """

        # Encodings for A, B, C
        # A=0 -> "AA"
        # B=1 -> "BB"
        # C=2 -> "AB"
        # We'll also use last=3 to indicate "no previous block" (start).
        adjacency = [
            [False, True,  False],  # A -> (A?, B?, C?)
            [True,  False, True ],  # B -> (A?, B?, C?)
            [True,  False, True ]   # C -> (A?, B?, C?)
        ]

        from functools import lru_cache

        @lru_cache(None)
        def dp(a, b, c, last):
            # a, b, c are counts of A, B, C left
            # last in {0, 1, 2, 3}, where 3 = "start"
            best = 0

            # try A if we have any left
            if a > 0:
                if last == 3 or adjacency[last][0]:  # can we go from last -> A?
                    best = max(best, 1 + dp(a - 1, b, c, 0))

            # try B if we have any left
            if b > 0:
                if last == 3 or adjacency[last][1]:  # can we go from last -> B?
                    best = max(best, 1 + dp(a, b - 1, c, 1))

            # try C if we have any left
            if c > 0:
                if last == 3 or adjacency[last][2]:  # can we go from last -> C?
                    best = max(best, 1 + dp(a, b, c - 1, 2))

            return best

        # Each chosen block contributes length 2
        return dp(x, y, z, 3) * 2