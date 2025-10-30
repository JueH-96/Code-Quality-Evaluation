class Solution:
    def minimumDeletions(self, word: str, k: int) -> int:
        # Count the frequency of each character
        freq = {}
        for char in word:
            freq[char] = freq.get(char, 0) + 1
        
        # Sort the frequencies in ascending order
        sorted_freq = sorted(freq.values())
        
        min_deletions = float('inf')
        
        # Try each frequency as the minimum frequency
        for i in range(len(sorted_freq)):
            base_freq = sorted_freq[i]
            deletions = sum(sorted_freq[:i])  # Delete all characters with lower frequency
            
            # For higher frequencies, delete excess characters
            for j in range(i, len(sorted_freq)):
                if sorted_freq[j] > base_freq + k:
                    deletions += sorted_freq[j] - (base_freq + k)
            
            min_deletions = min(min_deletions, deletions)
        
        return min_deletions