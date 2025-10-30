class Solution:
    def shortestBeautifulSubstring(self, s: str, k: int) -> str:
        n = len(s)
        min_length = float('inf')
        result = ""
        
        for i in range(n):
            count_ones = 0
            for j in range(i, n):
                if s[j] == '1':
                    count_ones += 1
                
                if count_ones > k:
                    break
                
                if count_ones == k:
                    current_length = j - i + 1
                    current_substring = s[i:j + 1]
                    
                    if current_length < min_length:
                        min_length = current_length
                        result = current_substring
                    elif current_length == min_length:
                        result = min(result, current_substring)
        
        return result