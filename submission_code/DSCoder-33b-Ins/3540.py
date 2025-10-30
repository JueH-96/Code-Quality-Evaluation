class Solution:
    def stringHash(self, s: str, k: int) -> str:
        result = ""
        for i in range(0, len(s), k):
            substring = s[i:i+k]
            sum_hash = sum(ord(c) - ord('a') for c in substring)
            hashed_char = chr(sum_hash % 26 + ord('a'))
            result += hashed_char
        return result