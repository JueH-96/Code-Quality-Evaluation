class Solution:
    def getEncryptedString(self, s: str, k: int) -> str:
        encrypted = []
        for c in s:
            encrypted.append(chr((ord(c) - ord('a') + k) % 26 + ord('a')))
        return ''.join(encrypted)