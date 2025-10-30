class Solution:
    def shortestBeautifulSubstring(self, s: str, k: int) -> str:
        n = len(s)
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i+1] = prefix[i] + (1 if s[i] == '1' else 0)
        
        candidates = []
        for i in range(n):
            for j in range(i, n):
                cnt = prefix[j+1] - prefix[i]
                if cnt == k:
                    substr = s[i:j+1]
                    candidates.append(substr)
        
        if not candidates:
            return ""
        
        min_len = min(len(c) for c in candidates)
        filtered = [c for c in candidates if len(c) == min_len]
        return min(filtered)