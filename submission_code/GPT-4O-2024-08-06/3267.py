class Solution:
    def maximumLength(self, s: str) -> int:
        max_length = -1
        n = len(s)
        
        # Iterate over each character in the string
        for i in range(n):
            # Start counting the length of the special substring
            length = 1
            while i + length < n and s[i] == s[i + length]:
                length += 1
            
            # Check if this special substring occurs at least thrice
            if length >= 3:
                # Calculate the maximum length that can occur thrice
                max_possible_length = length // 3
                max_length = max(max_length, max_possible_length)
            
            # Move the index to the end of the current special substring
            i += length - 1
        
        return max_length