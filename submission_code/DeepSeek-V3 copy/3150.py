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
                    current_len = j - i + 1
                    if current_len < min_len:
                        min_len = current_len
                        result = s[i:j+1]
                    elif current_len == min_len:
                        candidate = s[i:j+1]
                        if candidate < result:
                            result = candidate
                    break
                elif count > k:
                    break
        
        return result