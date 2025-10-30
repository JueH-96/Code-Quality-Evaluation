class Solution:
    def stringHash(self, s: str, k: int) -> str:
        n = len(s)
        result = []
        
        for i in range(0, n, k):
            substring = s[i:i+k]
            hash_value = sum(ord(char) - ord('a') for char in substring) % 26
            result.append(chr(hash_value + ord('a')))
        
        return ''.join(result)