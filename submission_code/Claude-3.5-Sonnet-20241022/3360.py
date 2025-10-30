class Solution:
    def minimumDeletions(self, word: str, k: int) -> int:
        # Count frequency of each character
        freq = {}
        for c in word:
            freq[c] = freq.get(c, 0) + 1
        
        # Get list of frequencies
        freqs = sorted(freq.values())
        
        min_deletions = float('inf')
        
        # Try each frequency as base frequency
        for i in range(len(freqs)):
            base = freqs[i]
            curr_deletions = 0
            
            # For each frequency, calculate deletions needed
            for f in freqs:
                if f < base:
                    # Delete all chars with frequency < base
                    curr_deletions += f
                elif f > base + k:
                    # Delete excess chars above base + k
                    curr_deletions += f - (base + k)
            
            min_deletions = min(min_deletions, curr_deletions)
        
        return min_deletions