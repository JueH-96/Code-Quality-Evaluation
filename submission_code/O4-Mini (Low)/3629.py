class Solution:
    def lengthAfterTransformations(self, s: str, t: int) -> int:
        MOD = 10**9 + 7
        
        # dp[i] will hold the length contribution of the letter with index i after current number of transformations
        # index 0 = 'a', 1 = 'b', ..., 25 = 'z'
        dp = [1] * 26
        
        # Perform t transformations on the dp array
        for _ in range(t):
            new_dp = [0] * 26
            # for 'a' through 'y', they shift to the next letter
            for i in range(25):
                new_dp[i] = dp[i+1]
            # for 'z', it becomes "ab"
            new_dp[25] = (dp[0] + dp[1]) % MOD
            dp = new_dp
        
        # Sum up the contributions for each character in the original string
        result = 0
        for ch in s:
            idx = ord(ch) - ord('a')
            result = (result + dp[idx]) % MOD
        
        return result