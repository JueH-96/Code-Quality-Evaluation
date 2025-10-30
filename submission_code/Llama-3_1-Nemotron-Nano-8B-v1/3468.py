class Solution:
    def getEncryptedString(self, s: str, k: int) -> str:
        n = len(s)
        if n == 0:
            return ""
        encrypted = []
        for i in range(n):
            target = (i + k) % n
            encrypted.append(s[target])
        return ''.join(encrypted)