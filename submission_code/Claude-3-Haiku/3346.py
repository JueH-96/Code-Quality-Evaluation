class Solution:
    def getSmallestString(self, s: str, k: int) -> str:
        n = len(s)
        result = ['a'] * n
        k -= n
        
        i = n - 1
        while k > 0:
            result[i] = chr(ord('a') + min(25, k))
            k -= min(25, k)
            i -= 1
        
        return ''.join(result)