class Solution:
    def countSteppingNumbers(self, low: str, high: str) -> int:
        MOD = 10**9 + 7
        
        def countSteppingNumbersUpTo(s):
            n = len(s)
            memo = {}
            
            def dp(pos, prev_digit, tight, started):
                if pos == n:
                    return 1 if started else 0
                
                if (pos, prev_digit, tight, started) in memo:
                    return memo[(pos, prev_digit, tight, started)]
                
                limit = int(s[pos]) if tight else 9
                result = 0
                
                for digit in range(0, limit + 1):
                    new_tight = tight and (digit == limit)
                    new_started = started or (digit > 0)
                    
                    # If we haven't started, we can place 0 (leading zero)
                    if not started and digit == 0:
                        result = (result + dp(pos + 1, -1, new_tight, new_started)) % MOD
                    # If we have started, check stepping condition
                    elif started or digit > 0:
                        if prev_digit == -1 or abs(digit - prev_digit) == 1:
                            result = (result + dp(pos + 1, digit, new_tight, new_started)) % MOD
                
                memo[(pos, prev_digit, tight, started)] = result
                return result
            
            return dp(0, -1, True, False)
        
        def isSteppingNumber(s):
            if len(s) == 1:
                return True
            for i in range(1, len(s)):
                if abs(int(s[i]) - int(s[i-1])) != 1:
                    return False
            return True
        
        count_high = countSteppingNumbersUpTo(high)
        count_low_minus_1 = 0
        
        # Calculate low - 1
        low_int = int(low)
        if low_int > 1:
            low_minus_1 = str(low_int - 1)
            count_low_minus_1 = countSteppingNumbersUpTo(low_minus_1)
        
        return (count_high - count_low_minus_1) % MOD