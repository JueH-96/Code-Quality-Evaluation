class Solution:
    def stringHash(self, s: str, k: int) -> str:
        result = ""
        n = len(s)
        for i in range(0, n, k):
            substring = s[i:i+k]
            total = 0
            for char in substring:
                total += ord(char) - ord('a')
            hashedChar = total % 26
            result += chr(hashedChar + ord('a'))
        return result