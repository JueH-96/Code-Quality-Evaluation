class Solution:
    def getEncryptedString(self, s: str, k: int) -> str:
        n = len(s)
        result = []
        
        for i in range(n):
            # Find the index of the character k positions ahead (cyclically)
            new_index = (i + k) % n
            result.append(s[new_index])
        
        return ''.join(result)