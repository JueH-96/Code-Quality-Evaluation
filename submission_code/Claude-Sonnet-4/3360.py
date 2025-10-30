class Solution:
    def minimumDeletions(self, word: str, k: int) -> int:
        from collections import Counter
        
        # Count frequency of each character
        freq = Counter(word)
        frequencies = list(freq.values())
        
        if not frequencies:
            return 0
        
        frequencies.sort()
        min_deletions = float('inf')
        
        # Try each frequency as the minimum target frequency
        for i in range(len(frequencies)):
            min_target = frequencies[i]
            max_target = min_target + k
            deletions = 0
            
            for f in frequencies:
                if f < min_target:
                    # Delete all occurrences of characters with frequency < min_target
                    deletions += f
                elif f > max_target:
                    # Keep only max_target occurrences, delete the rest
                    deletions += f - max_target
            
            min_deletions = min(min_deletions, deletions)
        
        # Also try ranges where we reduce some frequencies to create a valid range
        for i in range(len(frequencies)):
            for j in range(i, len(frequencies)):
                # Try range [frequencies[i], frequencies[j]] if it's valid
                if frequencies[j] - frequencies[i] <= k:
                    deletions = 0
                    for f in frequencies:
                        if f < frequencies[i]:
                            deletions += f
                        elif f > frequencies[j]:
                            deletions += f - frequencies[j]
                    min_deletions = min(min_deletions, deletions)
        
        return min_deletions