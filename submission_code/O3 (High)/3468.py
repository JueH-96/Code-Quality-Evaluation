class Solution:
    def getEncryptedString(self, s: str, k: int) -> str:
        n = len(s)
        if n == 0:
            return s                     # Guard (though length ≥ 1 by constraints)
        
        k %= n                           # Reduce k to the string length
        # For every position i, take the character k places ahead cyclically.
        encrypted = [s[(i + k) % n] for i in range(n)]
        return ''.join(encrypted)