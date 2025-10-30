class Solution:
    def shortestBeautifulSubstring(self, s: str, k: int) -> str:
        ones = [i for i, char in enumerate(s) if char == '1']
        n = len(ones)
        if n < k:
            return ""
        
        min_len = float('inf')
        candidates = []
        
        for i in range(n - k + 1):
            start = ones[i]
            end = ones[i + k - 1]
            current_len = end - start + 1
            current_sub = s[start:end+1]
            
            if current_len < min_len:
                min_len = current_len
                candidates = [current_sub]
            elif current_len == min_len:
                candidates.append(current_sub)
        
        return min(candidates) if candidates else ""