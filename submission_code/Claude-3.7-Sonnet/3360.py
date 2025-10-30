class Solution:
    def minimumDeletions(self, word: str, k: int) -> int:
        # Count frequency of each character
        freq = {}
        for char in word:
            freq[char] = freq.get(char, 0) + 1
        
        # Get sorted list of frequencies
        freq_values = sorted(freq.values())
        min_deletions = float('inf')
        
        # Try each frequency as the lower bound
        for i in range(len(freq_values)):
            lower_bound = freq_values[i]
            upper_bound = lower_bound + k
            
            deletions = 0
            for f in freq_values:
                if f < lower_bound:
                    # Delete all occurrences of characters below lower bound
                    deletions += f
                elif f > upper_bound:
                    # Delete excess occurrences of characters above upper bound
                    deletions += f - upper_bound
            
            min_deletions = min(min_deletions, deletions)
        
        return min_deletions