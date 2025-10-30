class Solution:
    def maxDifference(self, s: str) -> int:
        from collections import Counter
        
        # Count the frequency of each character in the string
        frequency = Counter(s)
        
        max_odd = float('-inf')
        max_even = float('-inf')
        
        # Iterate through the frequency dictionary to find max odd and even frequencies
        for count in frequency.values():
            if count % 2 == 0:  # Even frequency
                max_even = max(max_even, count)
            else:  # Odd frequency
                max_odd = max(max_odd, count)
        
        # Calculate the maximum difference
        return max_odd - max_even