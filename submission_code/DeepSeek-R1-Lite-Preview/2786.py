class Solution:
    def longestSemiRepetitiveSubstring(self, s: str) -> int:
        n = len(s)
        if n == 0:
            return 0
        
        left = 0
        pairs = 0
        pair_positions = []
        max_length = 1  # At least one character is always present
        
        for right in range(1, n):
            if s[right] == s[right - 1]:
                pairs += 1
                pair_positions.append(right - 1)
            
            while pairs > 1:
                first_pair_start = pair_positions[0]
                left = first_pair_start + 2
                pair_positions.pop(0)
                pairs -= 1
            
            current_window_length = right - left + 1
            if current_window_length > max_length:
                max_length = current_window_length
        
        return max_length