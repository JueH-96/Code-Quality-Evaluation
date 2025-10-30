class Solution:
    def minimumDeletions(self, word: str, k: int) -> int:
        from collections import Counter
        
        # Count frequencies
        freq = Counter(word)
        frequencies = list(freq.values())
        
        # If we have only one unique character, no deletions needed
        if len(frequencies) <= 1:
            return 0
        
        # If the word is already k-special, no deletions needed
        if max(frequencies) - min(frequencies) <= k:
            return 0
        
        min_deletions = float('inf')
        
        # Try all possible minimum frequencies
        for min_freq in range(0, max(frequencies) + 1):
            max_freq = min_freq + k
            deletions = 0
            
            for f in frequencies:
                if f < min_freq:
                    # Delete all occurrences of this character
                    deletions += f
                elif f > max_freq:
                    # Delete excess occurrences
                    deletions += f - max_freq
            
            min_deletions = min(min_deletions, deletions)
        
        return min_deletions