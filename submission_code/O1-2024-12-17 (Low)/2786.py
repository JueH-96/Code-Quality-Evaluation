class Solution:
    def longestSemiRepetitiveSubstring(self, s: str) -> int:
        # We will use a sliding window approach.
        # Maintain a window [left..right] within which there is at most one pair of consecutive identical digits.
        
        n = len(s)
        if n == 1:
            return 1
        
        left = 0
        repeated_pairs = 0  # count how many pairs of consecutive identical digits in current window
        max_len = 1
        
        for right in range(1, n):
            # Check if current character is same as previous
            if s[right] == s[right - 1]:
                repeated_pairs += 1
            
            # If more than one repeated pair, shrink from the left
            while repeated_pairs > 1:
                if s[left] == s[left + 1]:
                    repeated_pairs -= 1
                left += 1
            
            max_len = max(max_len, right - left + 1)
        
        return max_len