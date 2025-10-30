class Solution:
    def getEncryptedString(self, s: str, k: int) -> str:
        n = len(s)
        effective_k = k % n
        return s[effective_k:] + s[:effective_k]