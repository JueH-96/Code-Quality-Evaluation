class Solution:
    def shortestBeautifulSubstring(self, s: str, k: int) -> str:
        n = len(s)
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i+1] = prefix[i] + (1 if s[i] == '1' else 0)
        
        min_len = float('inf')
        candidates = []
        
        for i in range(n):
            for j in range(i, n):
                current_ones = prefix[j+1] - prefix[i]
                if current_ones > k:
                    break
                if current_ones == k:
                    current_len = j - i + 1
                    substr = s[i:j+1]
                    if current_len < min_len:
                        min_len = current_len
                        candidates = [substr]
                    elif current_len == min_len:
                        candidates.append(substr)
        
        if not candidates:
            return ""
        return min(candidates)