class Solution:
    def stringHash(self, s: str, k: int) -> str:
        n = len(s)
        result = ""
        for i in range(0, n, k):
            substring = s[i:i+k]
            hash_sum = 0
            for char in substring:
                hash_sum += ord(char) - ord('a')
            hashed_char = hash_sum % 26
            result += chr(ord('a') + hashed_char)
        return result