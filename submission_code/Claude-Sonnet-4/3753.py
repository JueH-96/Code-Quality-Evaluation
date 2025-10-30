class Solution:
    def maxDifference(self, s: str) -> int:
        # Count frequency of each character
        freq = {}
        for char in s:
            freq[char] = freq.get(char, 0) + 1
        
        # Separate odd and even frequencies
        odd_freqs = []
        even_freqs = []
        
        for count in freq.values():
            if count % 2 == 1:
                odd_freqs.append(count)
            else:
                even_freqs.append(count)
        
        # Find max odd frequency and min even frequency
        max_odd = max(odd_freqs)
        min_even = min(even_freqs)
        
        return max_odd - min_even