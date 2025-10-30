class Solution:
    def maximumLength(self, s: str) -> int:
        max_length = -1
        
        # Check for each character from 'a' to 'z'
        for char in set(s):
            count = 0
            current_length = 0
            
            # Iterate through the string to find special substrings
            for i in range(len(s)):
                if s[i] == char:
                    current_length += 1
                else:
                    if current_length > 0:
                        count += 1
                    current_length = 0
            
            # Check for the last segment
            if current_length > 0:
                count += 1
            
            # If this character has at least 3 segments
            if count >= 3:
                max_length = max(max_length, current_length)
        
        return max_length if max_length > 0 else -1