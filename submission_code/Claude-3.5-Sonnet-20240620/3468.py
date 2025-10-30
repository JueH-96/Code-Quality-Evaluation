class Solution:
    def getEncryptedString(self, s: str, k: int) -> str:
        n = len(s)
        result = []
        for i, char in enumerate(s):
            # Calculate the new index after shifting k positions
            new_index = (i + k) % n
            # Append the character at the new index to the result
            result.append(s[new_index])
        # Join the characters to form the encrypted string
        return ''.join(result)