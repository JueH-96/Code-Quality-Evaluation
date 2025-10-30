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
        
        # Find maximum difference between odd and even frequencies
        max_diff = float('-inf')
        for odd in odd_freqs:
            for even in even_freqs:
                diff = odd - even
                max_diff = max(max_diff, diff)
        
        return max_diff