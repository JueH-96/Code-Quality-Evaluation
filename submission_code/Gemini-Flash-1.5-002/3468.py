class Solution:
    def getEncryptedString(self, s: str, k: int) -> str:
        encrypted_string = ""
        for char in s:
            encrypted_char = chr(((ord(char) - ord('a') + k) % 26) + ord('a'))
            encrypted_string += encrypted_char
        return encrypted_string