class Solution:
    def lengthAfterTransformations(self, s: str, t: int) -> int:
        MOD = 10**9 + 7
        prev = [1] * 26  # Initialize for 0 transformations
        
        for _ in range(t):
            curr = [0] * 26
            for c in range(26):
                if c == 25:
                    curr[c] = (prev[0] + prev[1]) % MOD
                else:
                    curr[c] = prev[c + 1] % MOD
            prev = curr
        
        total = 0
        for ch in s:
            idx = ord(ch) - ord('a')
            total = (total + prev[idx]) % MOD
        
        return total