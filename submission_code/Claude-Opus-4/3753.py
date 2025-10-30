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
            if count % 2 == 0:
                even_freqs.append(count)
            else:
                odd_freqs.append(count)
        
        # Find maximum difference
        # We want max(odd) - min(even) to maximize the difference
        max_diff = float('-inf')
        
        for odd in odd_freqs:
            for even in even_freqs:
                max_diff = max(max_diff, odd - even)
        
        return max_diff