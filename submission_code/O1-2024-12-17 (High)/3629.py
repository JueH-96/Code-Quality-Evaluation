class Solution:
    def lengthAfterTransformations(self, s: str, t: int) -> int:
        MOD = 10**9 + 7
        
        # dp[c] will hold the length of the string after applying
        # transformations t times, starting from the character c
        # where c = 0 corresponds to 'a', 25 corresponds to 'z'.
        dp = [1] * 26  # Base for 0 transformations
        
        # Build up dp for 1..t transformations
        for _ in range(t):
            next_dp = [0] * 26
            # For 'a'.. 'y' (i.e. c=0..24), c -> c+1
            for c in range(25):
                next_dp[c] = dp[c + 1]
            # For 'z' (c=25), 'z' -> "ab", so length is dp['a'] + dp['b']
            next_dp[25] = (dp[0] + dp[1]) % MOD
            dp = next_dp
        
        # Sum up the results for all characters in s
        result = 0
        for ch in s:
            result = (result + dp[ord(ch) - ord('a')]) % MOD
        
        return result