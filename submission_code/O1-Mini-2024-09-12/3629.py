class Solution:
    def lengthAfterTransformations(self, s: str, t: int) -> int:
        MOD = 10**9 + 7
        # Initialize dp for k=0: each character counts as 1
        dp = {chr(ord('a') + i): 1 for i in range(26)}
        
        for _ in range(t):
            new_dp = {}
            for i in range(26):
                c = chr(ord('a') + i)
                if c != 'z':
                    next_c = chr(ord(c) + 1)
                    new_dp[c] = dp[next_c]
                else:
                    # 'z' becomes 'a' and 'b'
                    new_dp[c] = (dp['a'] + dp['b']) % MOD
            dp = new_dp
        
        total_length = 0
        for c in s:
            total_length = (total_length + dp[c]) % MOD
        return total_length