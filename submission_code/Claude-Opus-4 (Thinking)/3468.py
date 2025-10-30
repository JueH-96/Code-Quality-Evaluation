class Solution:
    def getEncryptedString(self, s: str, k: int) -> str:
        n = len(s)
        k = k % n  # Handle cases where k > n
        return s[k:] + s[:k]