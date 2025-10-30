class Solution:
    def lengthAfterTransformations(self, s: str, t: int) -> int:
        MOD = 10**9 + 7
        
        # dp[c] will hold the length of the string resulting from
        # transforming the character c ('a'->0, 'b'->1, ..., 'z'->25)
        # exactly t times.
        #
        # Base case (t=0): each character remains itself of length 1.
        dp = [1]*26
        
        for _ in range(t):
            new_dp = [0]*26
            # For characters 'a'..'y' (0..24), c -> c+1
            # So new_dp[c] = dp[c+1]
            for c in range(25):
                new_dp[c] = dp[c+1] % MOD
            
            # For 'z' (25), c -> "ab" => 'a' + 'b'
            # So new_dp[25] = dp['a'] + dp['b'] = dp[0] + dp[1]
            new_dp[25] = (dp[0] + dp[1]) % MOD
            
            dp = new_dp
        
        # Now sum up dp[value_of_letter] for each letter in s
        ans = 0
        for ch in s:
            ans = (ans + dp[ord(ch) - ord('a')]) % MOD
        
        return ans