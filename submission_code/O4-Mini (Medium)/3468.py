class Solution:
    def getEncryptedString(self, s: str, k: int) -> str:
        n = len(s)
        if n == 0:
            return ""
        # Reduce k modulo length of s for cyclic shift
        k %= n
        # For each position i, take the character at (i + k) % n
        return "".join(s[(i + k) % n] for i in range(n))