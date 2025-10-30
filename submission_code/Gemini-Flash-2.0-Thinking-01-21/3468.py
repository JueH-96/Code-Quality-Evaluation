class Solution:
    def getEncryptedString(self, s: str, k: int) -> str:
        encrypted_string = ""
        n = len(s)
        for i in range(n):
            new_index = (i + k) % n
            encrypted_string += s[new_index]
        return encrypted_string