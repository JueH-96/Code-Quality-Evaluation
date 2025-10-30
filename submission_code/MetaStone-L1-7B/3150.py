class Solution:
    def shortestBeautifulSubstring(self, s: str, k: int) -> str:
        n = len(s)
        if n == 0 or k == 0:
            return ""
        
        # Compute prefix sum array
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i+1] = prefix[i] + (s[i] == '1')
        
        min_len = float('inf')
        best_sub = ""
        
        for i in range(n):
            for j in range(i, n):
                total_ones = prefix[j+1] - prefix[i]
                if total_ones == k:
                    current_sub = s[i:j+1]
                    current_len = j - i + 1
                    if current_len < min_len:
                        min_len = current_len
                        best_sub = current_sub
                    elif current_len == min_len:
                        if current_sub < best_sub:
                            best_sub = current_sub
        
        return best_sub if min_len != float('inf') else ""