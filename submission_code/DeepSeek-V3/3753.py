class Solution:
    def maxDifference(self, s: str) -> int:
        frequency = {}
        for char in s:
            if char in frequency:
                frequency[char] += 1
            else:
                frequency[char] = 1
        
        odd_freqs = []
        even_freqs = []
        
        for count in frequency.values():
            if count % 2 == 1:
                odd_freqs.append(count)
            else:
                even_freqs.append(count)
        
        if not odd_freqs or not even_freqs:
            return 0  # though constraints say both exist
        
        max_odd = max(odd_freqs)
        min_even = min(even_freqs)
        
        return max_odd - min_even