class Solution:
    def countSteppingNumbers(self, low: str, high: str) -> int:
        MOD = 10**9 + 7
        
        def count_up_to(s):
            if not s or s == "-1":
                return 0
            
            n = len(s)
            memo = {}
            
            def dp(pos, tight, prev, started):
                if pos == n:
                    return 1 if started else 0
                
                if (pos, tight, prev, started) in memo:
                    return memo[(pos, tight, prev, started)]
                
                limit = int(s[pos]) if tight else 9
                result = 0
                
                for digit in range(0, limit + 1):
                    new_tight = tight and (digit == limit)
                    
                    if not started and digit == 0:
                        # Skip this digit (leading zero)
                        result = (result + dp(pos + 1, new_tight, prev, False)) % MOD
                    elif not started:
                        # Start the number with this digit
                        result = (result + dp(pos + 1, new_tight, digit, True)) % MOD
                    elif abs(digit - prev) == 1:
                        # Continue the stepping number
                        result = (result + dp(pos + 1, new_tight, digit, True)) % MOD
                
                memo[(pos, tight, prev, started)] = result
                return result
            
            return dp(0, True, -1, False)
        
        def subtract_one(s):
            if s == "0":
                return "-1"
            
            digits = list(s)
            i = len(digits) - 1
            
            # Handle borrowing from right to left
            while i >= 0 and digits[i] == '0':
                digits[i] = '9'
                i -= 1
            
            if i >= 0:
                digits[i] = str(int(digits[i]) - 1)
            
            # Remove leading zeros
            result = ''.join(digits).lstrip('0')
            return result if result else "0"
        
        count_high = count_up_to(high)
        low_minus_one = subtract_one(low)
        count_low_minus_one = count_up_to(low_minus_one)
        
        return (count_high - count_low_minus_one + MOD) % MOD