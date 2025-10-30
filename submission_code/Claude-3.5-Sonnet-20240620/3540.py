class Solution:
    def stringHash(self, s: str, k: int) -> str:
        result = []
        for i in range(0, len(s), k):
            substring = s[i:i+k]
            hash_sum = sum(ord(char) - ord('a') for char in substring)
            hashed_char = chr((hash_sum % 26) + ord('a'))
            result.append(hashed_char)
        return ''.join(result)