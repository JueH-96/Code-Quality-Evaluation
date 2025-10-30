class Solution:
    def getEncryptedString(self, s: str, k: int) -> str:
        n = len(s)
        encrypted_chars = [] # Use a list to build the result string efficiently
        
        # Iterate through each character position in the original string
        for i in range(n):
            # Calculate the new index for the character.
            # The character at the current position 'i' in the encrypted string
            # will be the character from the original string 's' located at
            # (i + k) positions away, treating the string as cyclic.
            new_index = (i + k) % n
            
            # Append the character from the original string 's' at the calculated new_index
            encrypted_chars.append(s[new_index])
            
        # Join the list of characters to form the final encrypted string
        return "".join(encrypted_chars)