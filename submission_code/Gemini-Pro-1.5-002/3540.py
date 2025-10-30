class Solution:
    def stringHash(self, s: str, k: int) -> str:
        n = len(s)
        result = ""
        for i in range(0, n, k):
            substring = s[i:i+k]
            sum_hash = 0
            for char in substring:
                sum_hash += ord(char) - ord('a')
            hashedChar = sum_hash % 26
            result += chr(ord('a') + hashedChar)
        return result