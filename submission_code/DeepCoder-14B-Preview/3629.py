MOD = 10**9 + 7

class Solution:
    def lengthAfterTransformations(self, s: str, t: int) -> int:
        # Precompute the transformation for each character
        transform = {}
        for c in 'abcdefghijklmnopqrstuvwxyz':
            if c == 'z':
                transform[c] = ['a', 'b']
            else:
                transform[c] = [chr(ord(c) + 1)]
        
        # Initialize DP
        prev = [1] * 26  # For t=0, each character contributes 1
        if t == 0:
            return len(s) % MOD
        
        for step in range(1, t + 1):
            curr = [0] * 26
            for c in 'abcdefghijklmnopqrstuvwxyz':
                c_idx = ord(c) - ord('a')
                for d in transform[c]:
                    d_idx = ord(d) - ord('a')
                    curr[c_idx] = (curr[c_idx] + prev[d_idx]) % MOD
            prev = curr[:]
        
        # Sum the contributions of each character in the input string
        total = 0
        for char in s:
            c_idx = ord(char) - ord('a')
            total = (total + prev[c_idx]) % MOD
        return total % MOD