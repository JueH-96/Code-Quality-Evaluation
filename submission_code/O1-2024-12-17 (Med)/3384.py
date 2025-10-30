class Solution:
    def minimumOperationsToMakeKPeriodic(self, word: str, k: int) -> int:
        from collections import Counter
        
        n = len(word)
        # Partition the string into blocks of length k
        blocks = [word[i:i + k] for i in range(0, n, k)]
        
        # Count frequencies of each distinct block
        freq = Counter(blocks)
        
        # We want all blocks to match one chosen block (the most frequent one)
        # The minimum operations = number_of_blocks - max_frequency_among_blocks
        return len(blocks) - max(freq.values())