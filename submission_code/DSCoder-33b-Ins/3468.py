class Solution:
    def getEncryptedString(self, s: str, k: int) -> str:
        encrypted_string = ""
        for c in s:
            encrypted_string += chr(((ord(c) - ord('a') + k) % 26) + ord('a'))
        return encrypted_string