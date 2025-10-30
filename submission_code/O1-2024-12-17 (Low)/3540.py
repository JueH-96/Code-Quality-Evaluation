class Solution:
    def stringHash(self, s: str, k: int) -> str:
        result = ""
        
        for i in range(0, len(s), k):
            substring = s[i : i + k]
            total = sum(ord(char) - ord('a') for char in substring)
            hashed_value = total % 26
            result += chr(hashed_value + ord('a'))
        
        return result