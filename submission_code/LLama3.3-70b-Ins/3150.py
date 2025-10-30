class Solution:
    def shortestBeautifulSubstring(self, s: str, k: int) -> str:
        n = len(s)
        min_len = float('inf')
        res = ""
        
        for i in range(n):
            for j in range(i + 1, n + 1):
                substring = s[i:j]
                if substring.count('1') == k:
                    if len(substring) < min_len:
                        min_len = len(substring)
                        res = substring
                    elif len(substring) == min_len and substring < res:
                        res = substring
        
        return res