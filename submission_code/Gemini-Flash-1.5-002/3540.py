class Solution:
    def stringHash(self, s: str, k: int) -> str:
        n = len(s)
        result = ""
        for i in range(0, n, k):
            substring = s[i:i+k]
            substring_sum = sum(ord(char) - ord('a') for char in substring)
            hashedChar = substring_sum % 26
            result += chr(ord('a') + hashedChar)
        return result