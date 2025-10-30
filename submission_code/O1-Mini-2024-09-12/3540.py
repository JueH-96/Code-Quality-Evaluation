class Solution:
    def stringHash(self, s: str, k: int) -> str:
        result = ""
        for i in range(0, len(s), k):
            substring = s[i:i+k]
            sum_values = sum(ord(c) - ord('a') for c in substring)
            hashed_value = sum_values % 26
            hashed_char = chr(ord('a') + hashed_value)
            result += hashed_char
        return result