class Solution:
    def shortestBeautifulSubstring(self, s: str, k: int) -> str:
        min_len = len(s) + 1
        res = ""
        n = len(s)
        for L in range(k, n + 1):
            for i in range(n - L + 1):
                substring = s[i:i + L]
                cnt1 = substring.count('1')
                if cnt1 == k:
                    if L < min_len:
                        min_len = L
                        res = substring
                    elif L == min_len and substring < res:
                        res = substring
        return res