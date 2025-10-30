class Solution:
    def numberOfBeautifulIntegers(self, low: int, high: int, k: int) -> int:
        # Helper: count beautiful numbers <= n
        def count_upto(n: int) -> int:
            if n < 0:
                return 0
            s = str(n)
            L = len(s)
            from functools import lru_cache
            @lru_cache(maxsize=None)
            def dp(pos: int, tight: bool, started: bool, even: int, odd: int, rem: int) -> int:
                # Base case: when we processed all digits
                if pos == L:
                    # Only count if we have started and have equal even and odd digits, and divisible by k.
                    if started and even == odd and rem == 0:
                        return 1
                    return 0
                ans = 0
                # Determine the maximum digit allowed at this position.
                limit = int(s[pos]) if tight else 9
                for d in range(0, limit + 1):
                    new_tight = tight and (d == limit)
                    if not started:
                        # We haven't started the number. If d==0, we can continue skipping.
                        if d == 0:
                            ans += dp(pos + 1, new_tight, False, even, odd, rem)
                        else:
                            # Start the number with digit d.
                            if d % 2 == 0:
                                new_even = even + 1
                                new_odd = odd
                            else:
                                new_even = even
                                new_odd = odd + 1
                            new_rem = (rem * 10 + d) % k
                            ans += dp(pos + 1, new_tight, True, new_even, new_odd, new_rem)
                    else:
                        # Already started: include digit d (even if it is zero).
                        if d % 2 == 0:
                            new_even = even + 1
                            new_odd = odd
                        else:
                            new_even = even
                            new_odd = odd + 1
                        new_rem = (rem * 10 + d) % k
                        ans += dp(pos + 1, new_tight, True, new_even, new_odd, new_rem)
                return ans
            return dp(0, True, False, 0, 0, 0)
        
        return count_upto(high) - count_upto(low - 1)