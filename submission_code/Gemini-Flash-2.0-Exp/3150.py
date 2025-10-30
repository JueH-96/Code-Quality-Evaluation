class Solution:
    def shortestBeautifulSubstring(self, s: str, k: int) -> str:
        n = len(s)
        min_len = float('inf')
        result = ""
        
        for i in range(n):
            for j in range(i, n):
                sub = s[i:j+1]
                count = sub.count('1')
                if count == k:
                    if len(sub) < min_len:
                        min_len = len(sub)
                        result = sub
                    elif len(sub) == min_len:
                        if sub < result:
                            result = sub
        
        if min_len == float('inf'):
            return ""
        else:
            return result