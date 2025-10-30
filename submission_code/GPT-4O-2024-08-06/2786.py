class Solution:
    def longestSemiRepetitiveSubstring(self, s: str) -> int:
        max_length = 1
        current_length = 1
        consecutive_pairs = 0
        
        for i in range(1, len(s)):
            if s[i] == s[i - 1]:
                consecutive_pairs += 1
                if consecutive_pairs > 1:
                    # Reset the current length to the length of the last valid semi-repetitive substring
                    current_length = i - last_pair_index
                    consecutive_pairs = 1
                last_pair_index = i
            else:
                consecutive_pairs = 0
            
            current_length += 1
            max_length = max(max_length, current_length)
        
        return max_length