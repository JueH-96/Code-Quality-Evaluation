class Solution:
    def shortestBeautifulSubstring(self, s: str, k: int) -> str:
        n = len(s)
        min_len = float('inf')
        result = ""
        
        for i in range(n):
            count_ones = 0
            for j in range(i, n):
                if s[j] == '1':
                    count_ones += 1
                if count_ones == k:
                    current_len = j - i + 1
                    current_substring = s[i:j+1]
                    if current_len < min_len or (current_len == min_len and current_substring < result):
                        min_len = current_len
                        result = current_substring
                    break
        
        return result