class Solution:
    def stringHash(self, s: str, k: int) -> str:
        result = []
        for i in range(0, len(s), k):
            substring = s[i:i + k]
            sum_hash = sum(ord(char) - ord('a') for char in substring)
            hashed_char = sum_hash % 26
            hashed_char_str = chr(hashed_char + ord('a'))
            result.append(hashed_char_str)
        return ''.join(result)