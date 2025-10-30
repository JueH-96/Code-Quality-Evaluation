class Solution:
    def minimumOperationsToMakeKPeriodic(self, word: str, k: int) -> int:
        n = len(word)
        groups = [[] for _ in range(k)]
        
        # Group characters by their position mod k
        for i in range(n):
            groups[i % k].append(word[i])
            
        total_ops = 0
        
        # For each group, find minimum operations to make all chars same
        for group in groups:
            # Count frequency of each char in group
            freq = {}
            for c in group:
                freq[c] = freq.get(c, 0) + 1
                
            # Find most frequent char
            max_freq = 0
            for f in freq.values():
                max_freq = max(max_freq, f)
                
            # Add operations needed for this group
            total_ops += len(group) - max_freq
            
        return total_ops