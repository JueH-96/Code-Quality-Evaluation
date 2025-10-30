class Solution:
    def longestSemiRepetitiveSubstring(self, s: str) -> int:
        max_length = 1
        left = 0
        last_pair_pos = -1  # Position of the last consecutive pair (i, i+1)
        
        for right in range(1, len(s)):
            if s[right] == s[right - 1]:
                current_pair_pos = right - 1  # Left index of the current pair
                # Check if the current pair is within the current window and there was a previous pair
                if last_pair_pos != -1 and current_pair_pos >= left:
                    # Move the left pointer to the position after the previous pair's start
                    left = last_pair_pos + 2
                # Update the last pair position
                last_pair_pos = current_pair_pos
            # Update the maximum length found
            current_length = right - left + 1
            if current_length > max_length:
                max_length = current_length
        return max_length