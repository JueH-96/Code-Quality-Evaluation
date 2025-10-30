class Solution:
    def getEncryptedString(self, s: str, k: int) -> str:
        n = len(s)
        encrypted_string = ""
        for i in range(n):
            current_char = s[i]
            replacement_index = (i + k) % n
            replacement_char = s[replacement_index]
            encrypted_string += replacement_char
        return encrypted_string