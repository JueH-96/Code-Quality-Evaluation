class Solution:
    def getEncryptedString(self, s: str, k: int) -> str:
        encrypted = []
        n = len(s)
        for c in s:
            # Calculate the new position in a cyclic manner
            original_index = ord(c) - ord('a')
            new_index = (original_index + k) % 26
            new_char = chr(new_index + ord('a'))
            encrypted.append(new_char)
        return ''.join(encrypted)