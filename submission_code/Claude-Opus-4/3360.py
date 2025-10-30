class Solution:
    def minimumDeletions(self, word: str, k: int) -> int:
        from collections import Counter
        
        # Count frequency of each character
        freq = Counter(word)
        frequencies = list(freq.values())
        
        # If all frequencies already satisfy the condition, no deletions needed
        if max(frequencies) - min(frequencies) <= k:
            return 0
        
        min_deletions = float('inf')
        
        # Try each possible minimum frequency
        # The minimum frequency can be 0 or any of the existing frequencies
        possible_mins = [0] + frequencies
        
        for min_freq in possible_mins:
            max_freq = min_freq + k
            deletions = 0
            
            for f in frequencies:
                if f < min_freq:
                    # Delete all occurrences of this character
                    deletions += f
                elif f > max_freq:
                    # Delete excess occurrences to bring it down to max_freq
                    deletions += f - max_freq
            
            min_deletions = min(min_deletions, deletions)
        
        return min_deletions