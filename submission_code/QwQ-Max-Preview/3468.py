class Solution:
    def getEncryptedString(self, s: str, k: int) -> str:
        n = len(s)
        encrypted = []
        for i in range(n):
            new_i = (i + k) % n
            encrypted.append(s[new_i])
        return ''.join(encrypted)