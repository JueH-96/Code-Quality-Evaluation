class Solution:
    def getEncryptedString(self, s: str, k: int) -> str:
        alphabet = 'abcdefghijklmnopqrstuvwxyz'
        encrypted_string = ''
        for c in s:
            index = alphabet.index(c)
            encrypted_string += alphabet[(index + k) % 26]
        return encrypted_string