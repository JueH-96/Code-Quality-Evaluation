class Solution:
    def getEncryptedString(self, s: str, k: int) -> str:
        n = len(s)
        result = []
        
        for i in range(n):
            # Get index k positions ahead in cyclic manner
            new_idx = (i + k) % n
            result.append(s[new_idx])
            
        return ''.join(result)