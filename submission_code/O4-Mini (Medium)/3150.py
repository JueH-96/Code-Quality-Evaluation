class Solution:
    def shortestBeautifulSubstring(self, s: str, k: int) -> str:
        # Collect indices of '1's
        ones = [i for i, ch in enumerate(s) if ch == '1']
        # If there are fewer than k ones, no beautiful substring exists
        if len(ones) < k:
            return ""
        
        n = len(s)
        min_len = n + 1
        candidates = []
        
        # Slide over windows of k consecutive ones
        for i in range(len(ones) - k + 1):
            left = ones[i]
            right = ones[i + k - 1]
            curr_len = right - left + 1
            substr = s[left:right + 1]
            
            if curr_len < min_len:
                # Found a shorter beautiful substring
                min_len = curr_len
                candidates = [substr]
            elif curr_len == min_len:
                # Same length, add to candidates
                candidates.append(substr)
        
        # Return the lexicographically smallest among the shortest
        return min(candidates) if candidates else ""