class Solution:
    def numberOfBeautifulIntegers(self, low: int, high: int, k: int) -> int:
        
        # We want to count how many numbers x in [low, high] satisfy:
        #   1) x % k == 0
        #   2) The number of even digits in x equals the number of odd digits in x
        #
        # Because high can be as large as 10^9, we cannot iterate directly over all numbers.
        # Instead, we use a "digit DP" approach to count how many valid numbers are <= a given n,
        # and then use that to find the result for [low, high] = count_up_to(high) - count_up_to(low - 1).
        #
        # Steps:
        # 1) Create a helper function "count_up_to(n)" that returns how many numbers <= n are "beautiful".
        # 2) Implement this using digit DP with states:
        #       - position (index in the digit array)
        #       - remainder mod k
        #       - difference between count of even and odd digits (will shift by an offset in DP storage)
        #       - tight (0 or 1), indicating if we're bound by the prefix of n
        # 3) The base case is when we've processed all digits; we check if remainder == 0 and diff == 0.

        def count_beautiful(n):
            """Return how many beautiful numbers are <= n."""
            if n < 0:
                return 0
            digits = list(map(int, str(n)))
            length = len(digits)

            # dp[pos][rem][diff+offset][tight] -> number of ways
            # - pos: current digit index (0-based from left)
            # - rem: current remainder mod k
            # - diff: difference between #evenDigits and #oddDigits so far (range -length..length)
            #   we store this with an offset so it fits in an array (offset = length)
            # - tight: whether we are still following the exact prefix of n up to pos

            from functools import lru_cache
            
            @lru_cache(None)
            def dfs(pos, rem, diff, tight):
                if pos == length:
                    # if we've placed all digits, check if remainder and diff are 0
                    return 1 if (rem == 0 and diff == 0) else 0

                limit = digits[pos] if tight else 9
                total_ways = 0
                for dig in range(limit + 1):
                    new_rem = (rem * 10 + dig) % k
                    # if digit is even, diff increases by 1, else decreases by 1
                    new_diff = diff + (1 if (dig % 2 == 0) else -1)
                    # when tight=1, it remains 1 only if dig == limit
                    new_tight = tight and (dig == limit)
                    # only proceed if absolute new_diff <= length - pos - 1 + ??? 
                    # but let's just store in the DP. We only handle new_diff from -length..length
                    if -length <= new_diff <= length:
                        total_ways += dfs(pos + 1, new_rem, new_diff, new_tight)
                return total_ways

            # Start with pos=0, rem=0, diff=0, tight=1
            return dfs(0, 0, 0, True)

        return count_beautiful(high) - count_beautiful(low - 1)