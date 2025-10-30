class Solution:
    def minimumDeletions(self, word: str, k: int) -> int:
        from collections import Counter
        
        # Count the frequency of each character in the word
        freq = Counter(word)
        
        # Get the frequencies in a sorted list
        freq_values = sorted(freq.values())
        
        # Initialize the minimum deletions to a large number
        min_deletions = float('inf')
        
        # Iterate over possible target frequencies
        for target_freq in range(freq_values[0], freq_values[-1] + 1):
            deletions = 0
            for f in freq_values:
                if f > target_freq + k:
                    # If the frequency is greater than target_freq + k, we need to delete some characters
                    deletions += f - (target_freq + k)
                elif f < target_freq - k:
                    # If the frequency is less than target_freq - k, we need to delete all characters
                    deletions += f
            
            # Update the minimum deletions
            min_deletions = min(min_deletions, deletions)
        
        return min_deletions