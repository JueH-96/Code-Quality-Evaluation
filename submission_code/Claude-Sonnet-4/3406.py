class Solution:
    def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
        MOD = 10**9 + 7
        
        # Memoization cache
        memo = {}
        
        def dp(zeros_left, ones_left, last_element, consecutive_count):
            # Base case: no more elements to place
            if zeros_left == 0 and ones_left == 0:
                return 1
            
            # Invalid state: ran out of one type but still have consecutive requirement
            if zeros_left < 0 or ones_left < 0:
                return 0
            
            # Check memo
            state = (zeros_left, ones_left, last_element, consecutive_count)
            if state in memo:
                return memo[state]
            
            result = 0
            
            # If we've hit the limit, we must switch to the other element
            if consecutive_count == limit:
                if last_element == 0:  # Must place a 1
                    if ones_left > 0:
                        result = dp(zeros_left, ones_left - 1, 1, 1)
                else:  # Must place a 0
                    if zeros_left > 0:
                        result = dp(zeros_left - 1, ones_left, 0, 1)
            else:
                # We can either continue with the same element or switch
                
                # Continue with same element
                if last_element == 0 and zeros_left > 0:
                    result += dp(zeros_left - 1, ones_left, 0, consecutive_count + 1)
                elif last_element == 1 and ones_left > 0:
                    result += dp(zeros_left, ones_left - 1, 1, consecutive_count + 1)
                
                # Switch to other element
                if last_element == 0 and ones_left > 0:
                    result += dp(zeros_left, ones_left - 1, 1, 1)
                elif last_element == 1 and zeros_left > 0:
                    result += dp(zeros_left - 1, ones_left, 0, 1)
                
                result %= MOD
            
            memo[state] = result
            return result
        
        # Start with either 0 or 1 as the first element
        total = 0
        
        # Start with 0
        if zero > 0:
            total += dp(zero - 1, one, 0, 1)
        
        # Start with 1
        if one > 0:
            total += dp(zero, one - 1, 1, 1)
        
        return total % MOD