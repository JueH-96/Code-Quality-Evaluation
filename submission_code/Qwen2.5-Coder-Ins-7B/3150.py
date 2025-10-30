class Solution:
    def shortestBeautifulSubstring(self, s: str, k: int) -> str:
        n = len(s)
        min_len = float('inf')
        min_substring = ""
        count = 0
        left = 0
        
        for right in range(n):
            if s[right] == '1':
                count += 1
            
            while count == k:
                if right - left + 1 < min_len:
                    min_len = right - left + 1
                    min_substring = s[left:right+1]
                elif right - left + 1 == min_len:
                    min_substring = min(min_substring, s[left:right+1])
                
                if s[left] == '1':
                    count -= 1
                left += 1
        
        return min_substring