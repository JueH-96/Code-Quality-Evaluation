class Solution:
    def getEncryptedString(self, s: str, k: int) -> str:
        n = len(s)
        k_eff = k % n
        encrypted = ''.join(s[(i + k_eff) % n] for i in range(n))
        return encrypted