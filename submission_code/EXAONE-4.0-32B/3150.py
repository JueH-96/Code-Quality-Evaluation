class Solution:
    def shortestBeautifulSubstring(self, s: str, k: int) -> str:
        n = len(s)
        prefix = [0] * (n + 1)
        for i in range(1, n + 1):
            prefix[i] = prefix[i - 1] + (1 if s[i - 1] == '1' else 0)
        
        candidates = []
        for L in range(k, n + 1):
            current_candidates = []
            for start in range(0, n - L + 1):
                ones_count = prefix[start + L] - prefix[start]
                if ones_count == k:
                    current_candidates.append(s[start:start + L])
            if current_candidates:
                candidates = current_candidates
                break
        
        return min(candidates) if candidates else ""