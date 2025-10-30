class Solution:
    def shortestBeautifulSubstring(self, s: str, k: int) -> str:
        n = len(s)
        min_len = float('inf')
        result = ""
        
        count = 0
        left = 0
        
        for right in range(n):
            if s[right] == '1':
                count += 1
            
            while count == k:
                if right - left + 1 < min_len:
                    min_len = right - left + 1
                    result = s[left:right + 1]
                elif right - left + 1 == min_len:
                    result = min(result, s[left:right + 1])
                
                if s[left] == '1':
                    count -= 1
                left += 1
        
        return result