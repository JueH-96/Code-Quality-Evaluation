class Solution:
    def getEncryptedString(self, s: str, k: int) -> str:
        n = len(s)
        k_mod = k % n
        return s[k_mod:] + s[:k_mod]