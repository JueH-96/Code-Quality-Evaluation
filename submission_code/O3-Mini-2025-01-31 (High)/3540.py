class Solution:
    def stringHash(self, s: str, k: int) -> str:
        result = ""
        # Process the string in chunks of size k
        for i in range(0, len(s), k):
            substring = s[i:i+k]
            char_sum = sum(ord(char) - ord('a') for char in substring)
            hashed_value = char_sum % 26
            result += chr(ord('a') + hashed_value)
        return result