class Solution:
    def lengthAfterTransformations(self, s: str, t: int) -> int:
        MOD = 10**9 + 7
        prev = [1] * 26  # Initial state: each character contributes 1 after 0 transformations
        
        for _ in range(t):
            curr = [0] * 26
            for c in range(26):
                if c == 25:  # 'z' case
                    curr[c] = (prev[0] + prev[1]) % MOD
                else:
                    next_c = c + 1
                    curr[c] = prev[next_c] % MOD
            prev = curr
        
        total = 0
        for char in s:
            idx = ord(char) - ord('a')
            total = (total + prev[idx]) % MOD
        
        return total