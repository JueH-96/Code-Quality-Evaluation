class Solution:
    def getEncryptedString(self, s: str, k: int) -> str:
        n = len(s)
        # Since we are taking the k-th character after each position (cyclic),
        # the new character at position i is s[(i + k) % n].
        encrypted = []
        for i in range(n):
            encrypted.append(s[(i + k) % n])
        return "".join(encrypted)