class Solution:
    def countSteppingNumbers(self, low: str, high: str) -> int:
        """
        We use digit-DP (dynamic programming) to count how many stepping numbers (no leading zero,
        adjacent digits differ by exactly 1) are <= a given number. Then the answer for [low, high]
        is f(high) - f(low - 1), taking care of big integer subtraction of 1 from 'low'.
        
        The digit-DP state is dp(pos, last_digit, started, tight), where:
          - pos is the current digit index in the given limit.
          - last_digit is the digit chosen for position pos-1 (or -1 if we haven't chosen yet).
          - started indicates whether we've chosen any nonzero digit so far
            (to avoid counting leading zeros as valid starting digits).
          - tight indicates whether we are bound by the prefix of the limit or free to choose up to 9.
        """
        import sys
        sys.setrecursionlimit(10**7)
        
        MOD = 10**9 + 7
        
        # Helper to subtract 1 from a decimal string num. Assumes num > "0".
        def subtract_one(num: str) -> str:
            if num == "0":
                return "0"  # not expected to happen in normal usage, but for safety.
            num_list = list(num)
            i = len(num_list) - 1
            while i >= 0:
                if num_list[i] == '0':
                    num_list[i] = '9'
                    i -= 1
                else:
                    num_list[i] = str(int(num_list[i]) - 1)
                    break
            # remove leading zeros if any
            while len(num_list) > 1 and num_list[0] == '0':
                num_list.pop(0)
            return "".join(num_list)
        
        def count_stepping_up_to(num: str) -> int:
            # Edge case: if num == "0", count of stepping numbers <= 0 is 0.
            if num == "0":
                return 0
            
            digits = list(map(int, num))
            n = len(digits)
            
            from functools import lru_cache
            
            @lru_cache(None)
            def dp(pos, last_digit, started, tight):
                # If we've reached the end of the number
                if pos == n:
                    # If we started picking digits (nonzero) at least once,
                    # we have a valid stepping number
                    return 1 if started else 0
                
                limit = digits[pos] if tight else 9
                total_ways = 0
                for dig in range(limit + 1):
                    # next_tight if we match the limit exactly at this position
                    next_tight = (tight and (dig == limit))
                    
                    # If we haven't started, and dig == 0, we continue not started
                    # But once we pick a non-zero digit, started becomes True
                    if not started:
                        # We can still pick zero and not be started yet,
                        # but that doesn't contribute to a valid stepping number
                        # unless eventually we pick a non-zero digit later.
                        if dig == 0:
                            total_ways += dp(pos + 1, -1, False, next_tight)
                        else:
                            # We start with a non-zero digit
                            total_ways += dp(pos + 1, dig, True, next_tight)
                    else:
                        # We have already started, so we must enforce stepping difference
                        if last_digit == -1:
                            # Shouldn't actually happen here, because once started is True,
                            # last_digit can't be -1. But for safety:
                            total_ways += dp(pos + 1, dig, True, next_tight)
                        else:
                            # Check stepping condition
                            if abs(last_digit - dig) == 1:
                                total_ways += dp(pos + 1, dig, True, next_tight)
                                
                    total_ways %= MOD
                
                return total_ways
            
            return dp(0, -1, False, True) % MOD
        
        # Main logic:
        # Count stepping <= high minus stepping <= (low-1)
        # Convert low-1 carefully, if low > "0"
        
        f_high = count_stepping_up_to(high)
        
        # If low == "1", then low - 1 = "0", which has count = 0
        # If low == "0", not in our constraints; we assume low >= 1
        if low == "1":
            f_low_minus_1 = 0
        else:
            low_minus_one = subtract_one(low)
            f_low_minus_1 = count_stepping_up_to(low_minus_one)
        
        return (f_high - f_low_minus_1) % MOD