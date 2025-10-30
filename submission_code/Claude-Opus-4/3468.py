class Solution:
    def getEncryptedString(self, s: str, k: int) -> str:
        n = len(s)
        result = []
        
        for i in range(n):
            # Find the k-th character after current position in cyclic manner
            new_index = (i + k) % n
            result.append(s[new_index])
        
        return ''.join(result)