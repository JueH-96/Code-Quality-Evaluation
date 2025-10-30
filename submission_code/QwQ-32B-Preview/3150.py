class Solution:
    def shortestBeautifulSubstring(self, s: str, k: int) -> str:
        # Step 1: Find all positions of '1' in the string
        ones_positions = [i for i, char in enumerate(s) if char == '1']
        
        # Step 2: If the number of ones is less than k, return empty string
        if len(ones_positions) < k:
            return ""
        
        # Step 3: Initialize variables to track the smallest substring
        min_length = float('inf')
        result = ""
        
        # Step 4: Iterate through the list of positions with a window of size k
        for i in range(len(ones_positions) - k + 1):
            left = ones_positions[i]
            right = ones_positions[i + k - 1]
            substring = s[left:right + 1]
            
            # Step 5: Update the result if a smaller length is found
            # Or if the same length is found, but the substring is lexicographically smaller
            if len(substring) < min_length:
                min_length = len(substring)
                result = substring
            elif len(substring) == min_length:
                if substring < result:
                    result = substring
        
        return result