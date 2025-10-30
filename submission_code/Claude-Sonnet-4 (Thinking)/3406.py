class Solution:
    def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
        MOD = 10**9 + 7
        
        # Memoization
        memo = {}
        
        def dp(zeros_left, ones_left, last_element, consecutive_count):
            if zeros_left == 0 and ones_left == 0:
                return 1
            if zeros_left < 0 or ones_left < 0 or consecutive_count > limit:
                return 0
            
            key = (zeros_left, ones_left, last_element, consecutive_count)
            if key in memo:
                return memo[key]
            
            result = 0
            
            # Try placing a 0
            if zeros_left > 0:
                if last_element == 0:
                    # Continue the consecutive sequence of 0s
                    result += dp(zeros_left - 1, ones_left, 0, consecutive_count + 1)
                else:
                    # Start a new sequence of 0s
                    result += dp(zeros_left - 1, ones_left, 0, 1)
            
            # Try placing a 1
            if ones_left > 0:
                if last_element == 1:
                    # Continue the consecutive sequence of 1s
                    result += dp(zeros_left, ones_left - 1, 1, consecutive_count + 1)
                else:
                    # Start a new sequence of 1s
                    result += dp(zeros_left, ones_left - 1, 1, 1)
            
            result %= MOD
            memo[key] = result
            return result
        
        # Start with placing either a 0 or a 1
        result = 0
        if zero > 0:
            result += dp(zero - 1, one, 0, 1)
        if one > 0:
            result += dp(zero, one - 1, 1, 1)
        
        return result % MOD