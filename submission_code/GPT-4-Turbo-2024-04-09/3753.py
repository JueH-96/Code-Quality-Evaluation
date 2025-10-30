class Solution:
    def maxDifference(self, s: str) -> int:
        from collections import Counter
        
        # Count the frequency of each character in the string
        freq = Counter(s)
        
        # Separate characters by even and odd frequencies
        odd_freqs = []
        even_freqs = []
        
        for char, count in freq.items():
            if count % 2 == 0:
                even_freqs.append(count)
            else:
                odd_freqs.append(count)
        
        # If there are no odd or even frequency characters, return 0
        if not odd_freqs or not even_freqs:
            return 0
        
        # Find the maximum difference
        max_diff = float('-inf')
        for odd in odd_freqs:
            for even in even_freqs:
                max_diff = max(max_diff, odd - even)
        
        return max_diff