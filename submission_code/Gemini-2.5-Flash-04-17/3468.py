class Solution:
    def getEncryptedString(self, s: str, k: int) -> str:
        n = len(s)
        encrypted_string = []
        for i in range(n):
            # Calculate the new index cyclically
            new_index = (i + k) % n
            # Append the character from the original string at the new index
            encrypted_string.append(s[new_index])
        
        # Join the list of characters to form the final string
        return "".join(encrypted_string)