class Solution:
    def numberOfBeautifulIntegers(self, low: int, high: int, k: int) -> int:
        """
        We want to count integers in [low, high] that have:
          1) The same number of even and odd digits.
          2) Divisible by k.

        Because high can be up to 10^9, iterating over all numbers is not feasible.
        Instead, we use a digit-DP (Dynamic Programming) approach. We'll create a
        function that, given an integer x, computes how many integers from 1 up to x
        satisfy the "beauty" conditions. Then our final result will be:
            count_beautiful(high) - count_beautiful(low - 1).

        Outline of the DP:
        - We'll process the digits of x from most significant to least significant.
        - The state we keep track of includes:
            pos: current index in the digit array (0-based),
            used: how many digits we've actually used so far (ignoring leading zeros),
            diff: the difference count_even - count_odd so far,
            rem: the remainder mod k so far,
            tight: indicates whether we're bound by the prefix of x (i.e., we cannot
                   exceed the current digit if tight == True).
        - Transitions:
            We pick the next digit "d" to place (from 0..9, or up to the digit at pos if tight).
            If used == 0 and d == 0, we remain in the "not-started" region (used=0, diff=0, rem=0).
            Once we place a nonzero digit (or if used > 0 already), we update:
                used -> used + 1 (still counting digits)
                diff -> diff + 1 if d is even, else diff - 1
                rem  -> (rem * 10 + d) % k
        - At the last position, we count a valid number only if:
            used > 0 (we actually formed a number),
            diff == 0 (equal even and odd digits),
            rem == 0  (divisible by k).

        The DP is cached (memoized) to avoid recomputation.

        This method ensures we only explore digits/paths that can form valid numbers
        up to x, without having to iterate through all possible integers up to x.
        """

        def digits_of(n: int) -> list:
            return list(map(int, str(n)))

        def count_beautiful(x: int) -> int:
            # Returns how many integers in [1..x] satisfy the beauty condition.
            # If x < 1, return 0.
            if x < 1:
                return 0

            digs = digits_of(x)
            n = len(digs)

            from functools import lru_cache

            @lru_cache(None)
            def dp(pos: int, used: int, diff: int, rem: int, tight: bool) -> int:
                # pos: current digit position (0-based),
                # used: how many digits have been used (non-leading zeros),
                # diff: (count_even - count_odd) so far,
                # rem: current remainder mod k,
                # tight: whether we are bound by the prefix of x at pos.
                if pos == n:
                    # If we've used at least 1 digit, and diff == 0, rem == 0, it's a valid number
                    return 1 if (used > 0 and diff == 0 and rem == 0) else 0

                limit = digs[pos] if tight else 9
                total_count = 0
                for d in range(limit + 1):
                    new_tight = tight and (d == limit)

                    # If we haven't started forming a number yet, and we choose 0,
                    # then we stay in used=0, diff=0, rem=0.
                    if used == 0 and d == 0:
                        total_count += dp(pos + 1, 0, 0, 0, new_tight)
                    else:
                        # If we've started or we are about to start with a nonzero digit
                        new_used = used + 1
                        # Update difference count
                        if d % 2 == 0:
                            # even digit
                            new_diff = diff + 1
                        else:
                            # odd digit
                            new_diff = diff - 1

                        new_rem = (rem * 10 + d) % k
                        total_count += dp(pos + 1, new_used, new_diff, new_rem, new_tight)

                return total_count

            return dp(0, 0, 0, 0, True)

        # Final count is numbers up to 'high' minus numbers up to 'low - 1'
        return count_beautiful(high) - count_beautiful(low - 1)