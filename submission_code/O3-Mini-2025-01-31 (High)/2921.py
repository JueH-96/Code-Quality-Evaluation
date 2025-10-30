class Solution:
    def countSteppingNumbers(self, low: str, high: str) -> int:
        mod = 10**9 + 7

        # Helper function to subtract 1 from a non-negative number represented as a string.
        def dec_str(num_str: str) -> str:
            # Convert to list for easier modification.
            arr = list(num_str)
            i = len(arr) - 1
            # Subtract one using elementary school subtraction.
            while i >= 0 and arr[i] == '0':
                arr[i] = '9'
                i -= 1
            if i < 0:
                # This happens if num_str was "0". We'll just return "0".
                return "0"
            arr[i] = str(int(arr[i]) - 1)
            # Remove any leading zeros.
            result = "".join(arr).lstrip("0")
            return result if result != "" else "0"
        
        # Count all valid stepping numbers <= x, where x is given as a string.
        def count_up_to(x: str) -> int:
            digits = list(map(int, x))
            n = len(digits)
            from functools import lru_cache

            # dp(pos, last, tight) where:
            #   pos: current index in the digit list [0, n)
            #   last: the last chosen digit; -1 indicates that we have not started the number yet.
            #   tight: whether the prefix is equal to the prefix of x.
            # When pos==n, if we have started (last != -1) then we have a valid constructed number.
            @lru_cache(maxsize=None)
            def dp(pos: int, last: int, tight: bool) -> int:
                if pos == n:
                    return 1 if last != -1 else 0  # Only count if we started (i.e. number is non-empty).
                
                res = 0
                # If tight, the maximum digit we can use at pos is digits[pos], otherwise it is 9.
                limit = digits[pos] if tight else 9

                if last == -1:
                    # We have not started our number yet. We have two kinds of options:
                    # 1. "Skip" this digit by choosing 0 (which means we still haven't started).
                    # 2. Choose a nonzero digit to start the number.
                    for d in range(0, limit + 1):
                        new_tight = tight and (d == limit)
                        if d == 0:
                            # Remain not started.
                            res += dp(pos + 1, -1, new_tight)
                        else:
                            # Start the number with digit d.
                            res += dp(pos + 1, d, new_tight)
                        res %= mod
                else:
                    # Already started. For stepping numbers, the next digit must differ by exactly 1.
                    # The only allowed choices are: last - 1 and last + 1 (if they lie in [0, 9]).
                    for d in (last - 1, last + 1):
                        if 0 <= d <= 9 and d <= limit:
                            new_tight = tight and (d == limit)
                            res += dp(pos + 1, d, new_tight)
                            res %= mod
                return res % mod

            return dp(0, -1, True)
        
        # Use inclusion–exclusion:
        # Count of stepping numbers in [low, high] = count_up_to(high) - count_up_to(low - 1).
        low_minus_one = dec_str(low)
        result = (count_up_to(high) - count_up_to(low_minus_one)) % mod
        return result