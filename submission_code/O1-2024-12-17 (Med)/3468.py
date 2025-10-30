class Solution:
    def getEncryptedString(self, s: str, k: int) -> str:
        n = len(s)
        # Reduce k to avoid unnecessary cycles
        k = k % n
        
        # Build the encrypted string
        encrypted = []
        for i in range(n):
            encrypted_char = s[(i + k) % n]
            encrypted.append(encrypted_char)
        
        return "".join(encrypted)