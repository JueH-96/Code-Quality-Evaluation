class Solution:
    def maxDifference(self, s: str) -> int:
        # Create a dictionary to store the frequency of each character
        freq = {}
        for char in s:
            freq[char] = freq.get(char, 0) + 1
        
        # Find the character with the maximum odd frequency and the character with the minimum even frequency
        max_odd = 0
        min_even = float('inf')
        for char, count in freq.items():
            if count % 2 == 1 and count > max_odd:
                max_odd = count
            elif count % 2 == 0 and count < min_even:
                min_even = count
        
        # Calculate the maximum difference
        if max_odd == 0 or min_even == float('inf'):
            return 0
        else:
            return max_odd - min_even