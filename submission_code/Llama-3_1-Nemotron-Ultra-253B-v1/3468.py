class Solution:
    def getEncryptedString(self, s: str, k: int) -> str:
        n = len(s)
        effective_k = k % n
        return ''.join([s[(i + effective_k) % n] for i in range(n)])