class Solution:
    def shortestBeautifulSubstring(self, s: str, k: int) -> str:
        n = len(s)
        min_len = float('inf')
        result = ""
        
        for start in range(n):
            count = 0
            for end in range(start, n):
                if s[end] == '1':
                    count += 1
                if count == k:
                    while s[start] == '0':
                        start += 1
                    temp = s[start:end+1]
                    if count == k and (min_len == len(temp) and temp < result or min_len > len(temp)):
                        min_len = len(temp)
                        result = temp
                    break
        
        return result