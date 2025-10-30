class Solution:
    def longestSemiRepetitiveSubstring(self, s: str) -> int:
        if len(s) <= 1:
            return len(s)
        
        left = 0
        max_length = 1
        last_pair_pos = -1  # Position where the last consecutive pair starts
        
        for right in range(1, len(s)):
            # Check if current character forms a consecutive pair with previous
            if s[right] == s[right - 1]:
                # If we already have a consecutive pair in our window
                if last_pair_pos != -1:
                    # Move left pointer to the position after the previous pair
                    left = last_pair_pos + 1
                # Update the position of the last consecutive pair
                last_pair_pos = right - 1
            
            # Update max length
            max_length = max(max_length, right - left + 1)
        
        return max_length