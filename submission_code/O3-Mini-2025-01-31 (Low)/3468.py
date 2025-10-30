class Solution:
    def getEncryptedString(self, s: str, k: int) -> str:
        n = len(s)
        # Since the transformation is cyclic, k might be reduced modulo n
        k = k % n
        encrypted = []
        for i in range(n):
            # Calculate the new index after shifting by k positions, cyclically.
            new_index = (i + k) % n
            encrypted.append(s[new_index])
        return "".join(encrypted)