class Solution:
    def getEncryptedString(self, s: str, k: int) -> str:
        n = len(s)
        encrypted_s = ""
        for i in range(n):
            encrypted_s += s[(i + k) % n]
        return encrypted_s