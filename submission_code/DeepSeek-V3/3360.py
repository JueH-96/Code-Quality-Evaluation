from collections import Counter

class Solution:
    def minimumDeletions(self, word: str, k: int) -> int:
        # Count the frequency of each character in the word
        freq = Counter(word)
        # Get the list of frequencies
        freq_list = sorted(freq.values(), reverse=True)
        min_deletions = float('inf')
        
        # Iterate over possible target frequencies
        for target in freq_list:
            current_deletions = 0
            for f in freq_list:
                if f > target + k:
                    current_deletions += f - (target + k)
                elif f < target:
                    current_deletions += f
            min_deletions = min(min_deletions, current_deletions)
        
        return min_deletions