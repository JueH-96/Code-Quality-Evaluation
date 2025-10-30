class Solution:
    def minimumDeletions(self, word: str, k: int) -> int:
        from collections import Counter
        
        # Count the frequency of each character in the word
        freq = Counter(word)
        # Extract the frequencies and sort them
        freq_values = sorted(freq.values())
        
        # Initialize the minimum deletions to a large number
        min_deletions = float('inf')
        
        # Try to make each frequency the minimum frequency in the k-special condition
        for i in range(len(freq_values)):
            deletions = 0
            min_freq = freq_values[i]
            max_allowed_freq = min_freq + k
            
            # Calculate deletions needed to make all frequencies within the range [min_freq, min_freq + k]
            for f in freq_values:
                if f < min_freq:
                    deletions += f
                elif f > max_allowed_freq:
                    deletions += f - max_allowed_freq
            
            # Update the minimum deletions found
            min_deletions = min(min_deletions, deletions)
        
        return min_deletions