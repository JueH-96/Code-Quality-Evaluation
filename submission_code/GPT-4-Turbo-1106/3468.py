class Solution:
    def getEncryptedString(self, s: str, k: int) -> str:
        encrypted = ""
        n = len(s)
        for i in range(n):
            encrypted += s[(i + k) % n]
        return encrypted