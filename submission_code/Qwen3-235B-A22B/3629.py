class Solution:
    def lengthAfterTransformations(self, s: str, t: int) -> int:
        MOD = 10**9 + 7
        prev = [1] * 26  # Initially, each character has count 1 at step 0
        
        for _ in range(t):
            curr = [0] * 26
            for c in range(26):
                if c == 25:  # 'z'
                    curr[c] = (prev[0] + prev[1]) % MOD
                else:
                    curr[c] = prev[c + 1]
            prev = curr  # Update prev to curr for the next iteration
        
        res = 0
        for ch in s:
            res = (res + prev[ord(ch) - ord('a')]) % MOD
        
        return res