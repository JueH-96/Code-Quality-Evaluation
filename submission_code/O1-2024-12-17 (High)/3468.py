class Solution:
    def getEncryptedString(self, s: str, k: int) -> str:
        n = len(s)
        # Since each character is replaced by the (i+k)-th character mod n, we can build the result directly
        encrypted_chars = []
        for i in range(n):
            encrypted_chars.append(s[(i + k) % n])
        return "".join(encrypted_chars)