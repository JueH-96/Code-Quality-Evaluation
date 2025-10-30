class Solution:
    def shortestBeautifulSubstring(self, s: str, k: int) -> str:
        # Initialize variables to track the shortest length and start index
        shortest_length = float('inf')
        start_index = -1
        
        # Initialize variables to use in the sliding window approach
        left = 0
        count_ones = 0
        
        # Iterate through the string with a sliding window
        for right in range(len(s)):
            # Count the number of 1's in the current window
            if s[right] == '1':
                count_ones += 1
            
            # If the window has k 1's, try to shrink it from the left
            while count_ones == k:
                # Update the shortest length and start index if the current window is smaller
                if right - left + 1 < shortest_length:
                    shortest_length = right - left + 1
                    start_index = left
                
                # Shrink the window from the left
                if s[left] == '1':
                    count_ones -= 1
                left += 1
        
        # If no beautiful substring was found, return an empty string
        if start_index == -1:
            return ""
        
        # Return the lexicographically smallest beautiful substring with length equal to shortest_length
        return s[start_index:start_index + shortest_length]

# Example usage:
# sol = Solution()
# print(sol.shortestBeautifulSubstring("100011001", 3))  # Output: "11001"
# print(sol.shortestBeautifulSubstring("1011", 2))       # Output: "11"
# print(sol.shortestBeautifulSubstring("000", 1))        # Output: ""