class Solution:
    def getEncryptedString(self, s: str, k: int) -> str:
        encrypted_string = ""
        for char in s:
            start = ord('a')
            offset = ord(char) - start
            new_offset = (offset + k) % 26
            encrypted_char = chr(start + new_offset)
            encrypted_string += encrypted_char
        return encrypted_string