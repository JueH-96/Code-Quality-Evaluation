class Solution:
    def numberOfBeautifulIntegers(self, low: int, high: int, k: int) -> int:
        # Helper function using digit DP to count beautiful numbers <= x.
        def count_beautiful(x: int) -> int:
            s = str(x)
            n = len(s)
            
            # dp(pos, tight, started, diff, rem) where:
            # pos: current index in the string (0-indexed)
            # tight: whether the prefix is equal to x's prefix (if True: can only place up to s[pos])
            # started: whether we have placed a non-zero digit before so that we count digits
            # diff: count(even) - count(odd) so far for the digits that are really placed (leading zeros not counted)
            # rem: current remainder mod k.
            from functools import lru_cache
            
            @lru_cache(maxsize=None)
            def dp(pos: int, tight: bool, started: bool, diff: int, rem: int) -> int:
                # Base case: At the end of the number
                if pos == n:
                    # Only consider if we've started and the counts are equal (diff == 0)
                    # and remainder mod k is 0.
                    if started and diff == 0 and rem == 0:
                        return 1
                    else:
                        return 0
                
                limit = int(s[pos]) if tight else 9
                total = 0
                for d in range(0, limit + 1):
                    new_tight = tight and (d == limit)
                    # Check if we have started: once started, even a 0 is counted as a placed digit.
                    new_started = started or (d != 0)
                    new_diff = diff
                    # If we are started, update diff: even digit adds +1, odd digit subtracts 1.
                    if new_started:
                        if d % 2 == 0:
                            new_diff += 1
                        else:
                            new_diff -= 1
                    new_rem = (rem * 10 + d) % k if new_started else rem
                    total += dp(pos + 1, new_tight, new_started, new_diff, new_rem)
                return total
            
            return dp(0, True, False, 0, 0)
        
        # Count numbers <= high and subtract numbers <= low-1.
        return count_beautiful(high) - count_beautiful(low - 1)


# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.numberOfBeautifulIntegers(10, 20, 3))  # Expected output: 2
    print(sol.numberOfBeautifulIntegers(1, 10, 1))   # Expected output: 1
    print(sol.numberOfBeautifulIntegers(5, 5, 2))    # Expected output: 0