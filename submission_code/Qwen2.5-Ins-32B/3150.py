class Solution:
    def shortestBeautifulSubstring(self, s: str, k: int) -> str:
        n = len(s)
        min_len = float('inf')
        result = ""
        
        for i in range(n):
            count = 0
            for j in range(i, n):
                if s[j] == '1':
                    count += 1
                if count == k:
                    if j - i + 1 < min_len:
                        min_len = j - i + 1
                        result = s[i:j+1]
                    elif j - i + 1 == min_len:
                        result = min(result, s[i:j+1])
                    break
        
        return result