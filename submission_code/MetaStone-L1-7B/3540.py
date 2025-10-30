class Solution:
    def stringHash(self, s: str, k: int) -> str:
        result = ""
        for i in range(0, len(s), k):
            substring = s[i:i+k]
            total = sum(ord(c) - ord('a') for c in substring)
            mod = total % 26
            result += chr(mod + ord('a'))
        return result