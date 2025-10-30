class Solution:
    def getEncryptedString(self, s: str, k: int) -> str:
        encrypted = ""
        for char in s:
            index = (ord(char) - ord('a') + k) % 26
            encrypted += chr(index + ord('a'))
        return encrypted