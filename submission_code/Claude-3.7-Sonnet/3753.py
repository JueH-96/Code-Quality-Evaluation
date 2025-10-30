class Solution:
    def maxDifference(self, s: str) -> int:
        # Count frequency of each character
        char_count = {}
        for char in s:
            char_count[char] = char_count.get(char, 0) + 1
        
        # Separate frequencies into odd and even groups
        odd_frequencies = []
        even_frequencies = []
        
        for count in char_count.values():
            if count % 2 == 0:
                even_frequencies.append(count)
            else:
                odd_frequencies.append(count)
        
        # Find maximum odd frequency and minimum even frequency
        max_odd = max(odd_frequencies)
        min_even = min(even_frequencies)
        
        # Return the difference
        return max_odd - min_even