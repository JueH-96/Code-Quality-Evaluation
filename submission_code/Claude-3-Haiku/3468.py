class Solution:
    def getEncryptedString(self, s: str, k: int) -> str:
        encrypted_string = ""
        for char in s:
            new_index = (ord(char) - ord('a') + k) % 26
            encrypted_string += chr(ord('a') + new_index)
        return encrypted_string