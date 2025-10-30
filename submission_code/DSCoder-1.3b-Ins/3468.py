class Solution:
    def getEncryptedString(self, s: str, k: int) -> str:
        encrypted_string = ""
        for i in range(len(s)):
            index = ord(s[i]) + k
            if index > ord('z'):
                index -= 26
            encrypted_string += chr(index)
        return encrypted_string