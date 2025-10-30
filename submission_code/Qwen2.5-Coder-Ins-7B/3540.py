class Solution:
    def stringHash(self, s: str, k: int) -> str:
        result = ""
        for i in range(0, len(s), k):
            substring = s[i:i+k]
            hash_value = sum(ord(char) - ord('a') for char in substring)
            hashedChar = hash_value % 26
            result += chr(ord('a') + hashedChar)
        return result