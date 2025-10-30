class Solution:
    def getEncryptedString(self, s: str, k: int) -> str:
        n = len(s)
        encrypted = []
        for i in range(n):
            encrypted.append(s[(i + k) % n])
        return ''.join(encrypted)