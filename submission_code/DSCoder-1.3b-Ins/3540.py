class Solution:
    def stringHash(self, s: str, k: int) -> str:
        result = ''
        n = len(s)
        for i in range(0, n, k):
            substring = s[i:i+k]
            sum_chars = sum([ord(c) - 96 for c in substring])
            hashed_char = sum_chars % 26
            result += chr(hashed_char + 97)
        return result