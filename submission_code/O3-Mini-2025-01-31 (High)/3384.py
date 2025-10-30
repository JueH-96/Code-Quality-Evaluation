class Solution:
    def minimumOperationsToMakeKPeriodic(self, word: str, k: int) -> int:
        # Divide the word into blocks (substrings) of length k.
        n = len(word)
        m = n // k  # number of blocks
        blocks = [word[i:i+k] for i in range(0, n, k)]
        
        # Count how many times each block appears.
        from collections import Counter
        count_blocks = Counter(blocks)
        
        # To make the word k-periodic, all blocks must be equal.
        # Since in one operation we can replace any block with any other, 
        # the best strategy is to pick the block that appears the most already,
        # and copy it over every block that is different.
        max_frequency = max(count_blocks.values())
        return m - max_frequency