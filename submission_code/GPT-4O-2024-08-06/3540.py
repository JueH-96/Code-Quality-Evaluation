class Solution:
    def stringHash(self, s: str, k: int) -> str:
        result = ""
        # Iterate over the string in chunks of size k
        for i in range(0, len(s), k):
            substring = s[i:i+k]
            # Calculate the sum of hash values of characters in the substring
            sum_hash = sum(ord(char) - ord('a') for char in substring)
            # Find the remainder when divided by 26
            hashedChar = sum_hash % 26
            # Convert hashedChar to the corresponding character
            result += chr(hashedChar + ord('a'))
        return result