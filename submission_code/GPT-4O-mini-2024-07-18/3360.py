class Solution:
    def minimumDeletions(self, word: str, k: int) -> int:
        from collections import Counter
        
        # Count frequency of each character in the word
        freq = Counter(word)
        frequencies = list(freq.values())
        
        # Sort frequencies to facilitate the calculation of deletions
        frequencies.sort()
        
        # The number of deletions needed
        min_deletions = float('inf')
        
        # Check each frequency as a target frequency
        for target_freq in frequencies:
            # Calculate the range of allowed frequencies
            low = target_freq
            high = target_freq + k
            
            # Count deletions needed to make all frequencies within [low, high]
            deletions = 0
            for f in frequencies:
                if f < low:
                    deletions += f  # Remove all of this frequency
                elif f > high:
                    deletions += f - high  # Remove excess to fit in the range
            
            # Update the minimum deletions found
            min_deletions = min(min_deletions, deletions)
        
        return min_deletions