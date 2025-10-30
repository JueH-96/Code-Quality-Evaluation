class Solution:
    def getEncryptedString(self, s: str, k: int) -> str:
        n = len(s)
        # Since the string is cyclic, we only need to shift by k % n
        k = k % n
        # Create the encrypted string by shifting each character by k positions
        encrypted_string = ""
        for i in range(n):
            # Find the new position for each character
            new_position = (i + k) % n
            encrypted_string += s[new_position]
        return encrypted_string