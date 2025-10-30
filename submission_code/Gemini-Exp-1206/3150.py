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
        
        
        final_result = ""
        for i in range(n):
            for j in range(i, n):
                sub = s[i:j+1]
                if len(sub) == min_len and sub.count('1') == k:
                    if final_result == "":
                        final_result = sub
                    elif sub < final_result:
                        final_result = sub
        return final_result