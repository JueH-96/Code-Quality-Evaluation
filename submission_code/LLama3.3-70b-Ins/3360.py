from collections import Counter

class Solution:
    def minimumDeletions(self, word: str, k: int) -> int:
        # Count the frequency of each character in the word
        freq = Counter(word)
        
        # Initialize the minimum and maximum frequencies
        min_freq = min(freq.values())
        max_freq = max(freq.values())
        
        # Initialize the minimum number of deletions
        min_deletions = float('inf')
        
        # Try all possible target frequencies
        for target_freq in range(min_freq, max_freq + 1):
            # Calculate the number of deletions needed to reach the target frequency
            deletions = sum(max(0, count - target_freq) for count in freq.values())
            
            # If the target frequency is k-special, update the minimum number of deletions
            if all(abs(count - target_freq) <= k for count in freq.values()):
                min_deletions = min(min_deletions, deletions)
        
        # Return the minimum number of deletions
        return min_deletions