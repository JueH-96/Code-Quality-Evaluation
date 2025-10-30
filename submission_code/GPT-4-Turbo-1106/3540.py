class Solution:
    def stringHash(self, s: str, k: int) -> str:
        result = ""
        for i in range(0, len(s), k):
            substring = s[i:i+k]
            sum_hashes = sum(ord(char) - ord('a') for char in substring)
            hashed_char = sum_hashes % 26
            result += chr(hashed_char + ord('a'))
        return result