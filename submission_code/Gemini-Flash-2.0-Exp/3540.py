class Solution:
    def stringHash(self, s: str, k: int) -> str:
        n = len(s)
        num_substrings = n // k
        result = ""
        for i in range(num_substrings):
            substring = s[i * k:(i + 1) * k]
            hash_sum = 0
            for char in substring:
                hash_sum += ord(char) - ord('a')
            hashed_char = hash_sum % 26
            result += chr(ord('a') + hashed_char)
        return result