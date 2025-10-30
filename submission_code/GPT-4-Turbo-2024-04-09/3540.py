class Solution:
    def stringHash(self, s: str, k: int) -> str:
        result = ""
        n = len(s)
        num_substrings = n // k
        
        for i in range(num_substrings):
            substring = s[i * k:(i + 1) * k]
            sum_hashes = sum(ord(char) - ord('a') for char in substring)
            hashed_char = sum_hashes % 26
            result += chr(hashed_char + ord('a'))
        
        return result