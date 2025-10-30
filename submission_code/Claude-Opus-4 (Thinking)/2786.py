class Solution:
    def longestSemiRepetitiveSubstring(self, s: str) -> int:
        n = len(s)
        if n <= 1:
            return n
        
        left = 0
        last_pair_idx = -1  # Index where the last consecutive pair starts
        max_length = 1
        
        for right in range(1, n):
            # Check if current character forms a consecutive pair
            if s[right] == s[right - 1]:
                # If we already have a consecutive pair, move left pointer
                if last_pair_idx != -1:
                    left = last_pair_idx + 1
                last_pair_idx = right - 1
            
            # Update maximum length
            max_length = max(max_length, right - left + 1)
        
        return max_length