class Solution:
    def countSteppingNumbers(self, low: str, high: str) -> int:
        MOD = 10**9 + 7
        
        # Check if a number is a stepping number
        def is_stepping_number(n: str) -> bool:
            if len(n) == 1:
                return True
            for i in range(1, len(n)):
                if abs(int(n[i]) - int(n[i-1])) != 1:
                    return False
            return True
        
        # Count stepping numbers less than or equal to a given number
        def count_stepping_numbers_le(n: str) -> int:
            memo = {}
            
            def dp(pos, tight, prev, started):
                # Base case: if we've gone through all positions
                if pos == len(n):
                    return 1 if started else 0
                
                # If we've already computed this state
                if (pos, tight, prev, started) in memo:
                    return memo[(pos, tight, prev, started)]
                
                result = 0
                limit = int(n[pos]) if tight else 9
                
                # If we haven't started placing digits
                if not started:
                    # We can place 0 (continue not placing)
                    result = (result + dp(pos + 1, tight and int(n[pos]) == 0, -1, False)) % MOD
                    
                    # Or start placing with 1 to limit
                    for d in range(1, limit + 1):
                        new_tight = tight and d == limit
                        result = (result + dp(pos + 1, new_tight, d, True)) % MOD
                else:
                    # If we've already started placing, we can place 0 to limit
                    for d in range(limit + 1):
                        if abs(d - prev) == 1:
                            new_tight = tight and d == limit
                            result = (result + dp(pos + 1, new_tight, d, True)) % MOD
                
                memo[(pos, tight, prev, started)] = result
                return result
            
            return dp(0, True, -1, False)
        
        # Count stepping numbers in the range [low, high]
        count_high = count_stepping_numbers_le(high)
        count_low_minus_1 = count_stepping_numbers_le(low) - (1 if is_stepping_number(low) else 0)
        
        return (count_high - count_low_minus_1) % MOD