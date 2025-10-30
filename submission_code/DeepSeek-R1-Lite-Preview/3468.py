class Solution:
    def getEncryptedString(self, s: str, k: int) -> str:
        encrypted_chars = [s[(i + k) % len(s)] for i in range(len(s))]
        return ''.join(encrypted_chars)