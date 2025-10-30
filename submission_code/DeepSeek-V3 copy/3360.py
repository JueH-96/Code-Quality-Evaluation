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
            deletions = 0
            for f in freq_list:
                if f > target + k:
                    deletions += f - (target + k)
                elif f < target:
                    deletions += f
            if deletions < min_deletions:
                min_deletions = deletions
        
        return min_deletions