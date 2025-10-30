class Solution:
    def stringHash(self, s: str, k: int) -> str:
        n = len(s)
        result = ""
        for i in range(0, n, k):
            substring = s[i:i+k]
            sum_hashes = sum(ord(c) - ord('a') for c in substring)
            hashedChar = sum_hashes % 26
            result += chr(hashedChar + ord('a'))
        return result