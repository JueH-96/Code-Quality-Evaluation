class Solution:
    def lengthAfterTransformations(self, s: str, t: int) -> int:
        MOD = 10**9 + 7
        
        # dp[c][i] will store the length of the string after i transformations,
        # starting from the character c (where c is an integer from 0 to 25 corresponding to 'a'..'z').
        dp = [[0] * (t + 1) for _ in range(26)]
        
        # Base case: if we do 0 transformations, the length is always 1
        for c in range(26):
            dp[c][0] = 1
        
        # Build up the DP table
        for i in range(1, t + 1):
            # For c != 'z' (which is 25 in 0-based index), the next char c+1 determines the length
            for c in range(25):
                dp[c][i] = dp[c + 1][i - 1]
            # For 'z', it becomes "ab", so dp[25][i] = dp['a'][i-1] + dp['b'][i-1]
            dp[25][i] = (dp[0][i - 1] + dp[1][i - 1]) % MOD
        
        # Sum up the contribution of each character in s
        result = 0
        for char in s:
            c_index = ord(char) - ord('a')
            result = (result + dp[c_index][t]) % MOD
        
        return result