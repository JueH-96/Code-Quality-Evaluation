class Solution:
    def getEncryptedString(self, s: str, k: int) -> str:
        len_s = len(s)
        effective_k = k % len_s
        return s[effective_k:] + s[:effective_k]