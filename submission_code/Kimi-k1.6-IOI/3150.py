class Solution:
    def shortestBeautifulSubstring(self, s: str, k: int) -> str:
        ones = [i for i, c in enumerate(s) if c == '1']
        n = len(ones)
        if n < k:
            return ""
        
        min_len = float('inf')
        # Find the minimum length among all possible windows
        for i in range(n - k + 1):
            current_len = ones[i + k - 1] - ones[i] + 1
            if current_len < min_len:
                min_len = current_len
        
        candidates = []
        # Collect all substrings with the minimum length
        for i in range(n - k + 1):
            start = ones[i]
            end = ones[i + k - 1]
            if (end - start + 1) == min_len:
                candidates.append(s[start:end+1])
        
        # Return the lexicographically smallest candidate
        return min(candidates) if candidates else ""