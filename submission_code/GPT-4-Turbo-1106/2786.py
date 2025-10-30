class Solution:
    def longestSemiRepetitiveSubstring(self, s: str) -> int:
        max_length = 0
        current_length = 0
        i = 0
        
        while i < len(s):
            # Start a new potential semi-repetitive substring
            current_length = 1
            has_pair = False
            
            for j in range(i + 1, len(s)):
                # If the current character is the same as the previous one
                if s[j] == s[j - 1]:
                    if has_pair:
                        # We already have a pair, so break the loop
                        break
                    else:
                        # Mark that we have found a pair
                        has_pair = True
                current_length += 1
            
            # Update the maximum length if the current one is longer
            max_length = max(max_length, current_length)
            # Move to the next character
            i += 1
        
        return max_length