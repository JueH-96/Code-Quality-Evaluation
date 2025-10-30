class Solution:
    def getEncryptedString(self, s: str, k: int) -> str:
        encrypted = []
        n = len(s)
        for char in s:
            # Calculate the new position in a cyclic manner
            original_pos = ord(char) - ord('a')
            new_pos = (original_pos + k) % 26
            new_char = chr(new_pos + ord('a'))
            encrypted.append(new_char)
        return ''.join(encrypted)