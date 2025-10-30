class Solution:
    def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
        MOD = 10**9 + 7
        
        # dp[i][j][k] represents number of ways to place i zeros and j ones
        # where k is length of current sequence of same digits
        @cache
        def dp(zeros, ones, curr_seq):
            # Base case - used all zeros and ones
            if zeros == 0 and ones == 0:
                return 1
                
            # Invalid case - sequence too long
            if curr_seq > limit:
                return 0
                
            ans = 0
            
            # Try placing 0
            if zeros > 0:
                if curr_seq > 0 and int(curr_seq > 0) == 1:
                    # Adding 0 after 1s
                    ans = (ans + dp(zeros-1, ones, 1)) % MOD
                else:
                    # Adding 0 after 0s
                    ans = (ans + dp(zeros-1, ones, curr_seq+1)) % MOD
            
            # Try placing 1        
            if ones > 0:
                if curr_seq > 0 and int(curr_seq > 0) == 0:
                    # Adding 1 after 0s
                    ans = (ans + dp(zeros, ones-1, 1)) % MOD
                else:
                    # Adding 1 after 1s
                    ans = (ans + dp(zeros, ones-1, curr_seq+1)) % MOD
                    
            return ans
            
        return dp(zero, one, 0)