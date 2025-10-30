class Solution:
    def stringHash(self, s: str, k: int) -> str:
        result = ""
        # Iterate over s in steps of k
        for i in range(0, len(s), k):
            substring = s[i:i+k]
            # Sum the hash values of characters in the substring
            total = 0
            for char in substring:
                total += (ord(char) - ord('a'))
            # Determine the hashed character
            hashed_char = total % 26
            # Convert to the corresponding character and append to result
            result += chr(hashed_char + ord('a'))
        return result