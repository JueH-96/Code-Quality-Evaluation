class Solution:
    def getEncryptedString(self, s: str, k: int) -> str:
        n = len(s)
        k = k % n  # Reduce k to avoid unnecessary full cycles
        encrypted = []
        
        for char in s:
            new_index = (ord(char) - ord('a') + k) % 26
            encrypted_char = chr(new_index + ord('a'))
            encrypted.append(encrypted_char)
        
        return ''.join(encrypted)