class Solution:
    def stringHash(self, s: str, k: int) -> str:
        result = ""
        n = len(s)
        num_substrings = n // k
        for i in range(num_substrings):
            substring = s[i * k : (i + 1) * k]
            substring_sum = 0
            for char in substring:
                substring_sum += ord(char) - ord('a')
            hashedChar = substring_sum % 26
            result += chr(ord('a') + hashedChar)
        return result