class Solution:
    def numberOfWays(self, s: str, t: str, k: int) -> int:
        MOD = 10**9 + 7
        n = len(s)
        
        # Function to compute the number of ways to transform s to t with exactly k moves
        def count_ways(s, t, k):
            if k == 0:
                return 1 if s == t else 0
            
            # Precompute all rotations of s
            rotations = [s[i:] + s[:i] for i in range(n)]
            
            # dp[j][i] will be the number of ways to reach rotation i using j moves
            dp = [[0] * n for _ in range(2)]
            dp[0][0] = 1  # Start with the original string s
            
            for step in range(1, k + 1):
                current = step % 2
                previous = 1 - current
                dp[current] = [0] * n
                for i in range(n):
                    # Sum all possible previous states that can rotate to this state
                    for l in range(1, n):  # l is the length of the suffix to rotate
                        prev_index = (i + l) % n
                        dp[current][i] = (dp[current][i] + dp[previous][prev_index]) % MOD
            
            # Find the index of t in rotations
            target_index = -1
            for i in range(n):
                if rotations[i] == t:
                    target_index = i
                    break
            
            return dp[k % 2][target_index] if target_index != -1 else 0
        
        return count_ways(s, t, k)