class Solution:
    def stringCount(self, n: int) -> int:
        MOD = 10**9 + 7
        
        # Initialize dp array
        dp = [0] * 16
        dp[0] = 1
        
        # Iterate n times
        for _ in range(n):
            new_dp = [0] * 16
            
            # Iterate through all possible states
            for state in range(16):
                # Add any character except l, e, t
                new_dp[state] = (new_dp[state] + dp[state] * 23) % MOD
                
                # Add 'l'
                if state & 1 == 0:
                    new_dp[state | 1] = (new_dp[state | 1] + dp[state]) % MOD
                
                # Add 'e'
                if state & 2 == 0:
                    new_dp[state | 2] = (new_dp[state | 2] + dp[state]) % MOD
                elif state & 4 == 0:
                    new_dp[state | 4] = (new_dp[state | 4] + dp[state]) % MOD
                
                # Add 't'
                if state & 8 == 0:
                    new_dp[state | 8] = (new_dp[state | 8] + dp[state]) % MOD
            
            dp = new_dp
        
        # Sum up all states that have 'l', 'e', 'e', 't'
        return sum(dp[15:]) % MOD