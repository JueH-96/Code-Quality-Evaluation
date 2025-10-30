class Solution:
    def minimumDeletions(self, word: str, k: int) -> int:
        counts = {}
        for char in word:
            counts[char] = counts.get(char, 0) + 1
        
        frequencies = sorted(counts.values())
        n = len(frequencies)
        
        min_deletions = float('inf')
        
        for i in range(n):
            deletions = 0
            target_freq = frequencies[i]
            
            for j in range(n):
                if frequencies[j] > target_freq + k:
                    deletions += frequencies[j] - (target_freq + k)
                
            min_deletions = min(min_deletions, deletions)
            
        return min_deletions