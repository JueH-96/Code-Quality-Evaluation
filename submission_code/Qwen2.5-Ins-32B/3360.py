class Solution:
    def minimumDeletions(self, word: str, k: int) -> int:
        from collections import Counter
        
        # Count the frequency of each character in the word
        freq = Counter(word)
        
        # Get the frequencies as a sorted list
        freq_values = sorted(freq.values(), reverse=True)
        
        # Initialize the minimum deletions to a large number
        min_deletions = float('inf')
        
        # Iterate over each frequency as a potential minimum frequency
        for min_freq in freq_values:
            deletions = 0
            for f in freq_values:
                if f < min_freq:
                    deletions += f
                else:
                    deletions += max(0, f - (min_freq + k))
            min_deletions = min(min_deletions, deletions)
        
        return min_deletions