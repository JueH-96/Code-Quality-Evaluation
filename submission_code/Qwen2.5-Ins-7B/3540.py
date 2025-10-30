class Solution:
    def stringHash(self, s: str, k: int) -> str:
        result = ""
        for i in range(0, len(s), k):
            substring = s[i:i+k]
            hash_sum = sum((ord(char) - ord('a')) for char in substring)
            hashedChar = hash_sum % 26
            result += chr(hashedChar + ord('a'))
        return result