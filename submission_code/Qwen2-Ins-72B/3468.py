class Solution:
    def getEncryptedString(self, s: str, k: int) -> str:
        n = len(s)
        encrypted = ""
        for i in range(n):
            # Calculate the index of the k^th character after the current character
            next_index = (i + k) % n
            encrypted += s[next_index]
        return encrypted