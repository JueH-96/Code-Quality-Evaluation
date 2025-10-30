class Solution:
    def stringHash(self, s: str, k: int) -> str:
        result = []
        for i in range(0, len(s), k):
            substring = s[i:i+k]
            total = sum(ord(c) - ord('a') for c in substring)
            hashed_char = chr(total % 26 + ord('a'))
            result.append(hashed_char)
        return ''.join(result)