class Solution:
    def stringHash(self, s: str, k: int) -> str:
        result = ""
        # iterate in steps of length k
        for i in range(0, len(s), k):
            substring = s[i:i+k]
            total = sum(ord(ch) - ord('a') for ch in substring)
            remainder = total % 26
            result += chr(remainder + ord('a'))
        return result