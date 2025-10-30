class Solution:
    def maxDifference(self, s: str) -> int:
        from collections import Counter
        
        # Count the frequency of each character in the string
        freq = Counter(s)
        
        # Initialize variables to store the maximum odd and even frequencies
        max_odd = float('-inf')
        max_even = float('-inf')
        
        # Iterate over the frequency dictionary
        for char, count in freq.items():
            if count % 2 == 0:
                # Update max_even if the current count is even and greater than max_even
                max_even = max(max_even, count)
            else:
                # Update max_odd if the current count is odd and greater than max_odd
                max_odd = max(max_odd, count)
        
        # Return the maximum difference between odd and even frequencies
        return max_odd - max_even