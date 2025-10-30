class Solution:
    def countSteppingNumbers(self, low: str, high: str) -> int:
        MOD = 10**9 + 7
        
        def countUpTo(num_str):
            if not num_str or num_str == "0":
                return 0
            
            n = len(num_str)
            # dp[pos][last_digit][tight][started]
            # pos: current position
            # last_digit: last digit placed (0-9)
            # tight: whether we're still bounded by the input number
            # started: whether we've started placing non-zero digits
            memo = {}
            
            def dp(pos, last_digit, tight, started):
                if pos == n:
                    return 1 if started else 0
                
                state = (pos, last_digit, tight, started)
                if state in memo:
                    return memo[state]
                
                limit = int(num_str[pos]) if tight else 9
                result = 0
                
                for digit in range(0, limit + 1):
                    if not started:
                        # Haven't started yet
                        if digit == 0:
                            # Still leading zeros
                            result += dp(pos + 1, -1, False, False)
                        else:
                            # Starting with this digit
                            result += dp(pos + 1, digit, tight and (digit == limit), True)
                    else:
                        # Already started, check stepping condition
                        if abs(digit - last_digit) == 1:
                            result += dp(pos + 1, digit, tight and (digit == limit), True)
                
                result %= MOD
                memo[state] = result
                return result
            
            return dp(0, -1, True, False)
        
        # Count stepping numbers from 1 to high
        count_high = countUpTo(high)
        
        # Count stepping numbers from 1 to low-1
        # Need to handle low-1 carefully
        low_int = int(low)
        if low_int == 1:
            count_low_minus_1 = 0
        else:
            low_minus_1 = str(low_int - 1)
            count_low_minus_1 = countUpTo(low_minus_1)
        
        return (count_high - count_low_minus_1) % MOD