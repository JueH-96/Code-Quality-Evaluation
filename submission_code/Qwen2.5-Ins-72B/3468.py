class Solution:
    def getEncryptedString(self, s: str, k: int) -> str:
        n = len(s)
        k = k % n  # To handle large k values
        encrypted = [''] * n
        for i in range(n):
            encrypted[i] = s[(i + k) % n]
        return ''.join(encrypted)