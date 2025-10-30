from collections import Counter

class Solution:
    def minimumOperationsToMakeKPeriodic(self, word: str, k: int) -> int:
        # Calculate the number of blocks the word is divided into.
        n = len(word)
        m = n // k

        # Count the frequency of each distinct block of length k.
        block_counts = Counter()
        for i in range(m):
            # Extract the i-th block (0-indexed).
            # The i-th block starts at index i * k and has length k.
            block = word[i * k : (i + 1) * k]
            block_counts[block] += 1

        # Find the maximum frequency among all blocks.
        # The constraints (1 <= n, k <= n, k divides n) ensure that
        # m = n // k is at least 1. Therefore, block_counts will contain
        # at least one entry, and block_counts.values() will not be empty.
        max_freq = max(block_counts.values())

        # The minimum number of operations needed is the total number of blocks
        # minus the frequency of the most common block. We can make all blocks
        # identical to the most common block type using one operation
        # for each block that is not already the most common one.
        # The number of blocks that are *not* the most frequent type is
        # the total number of blocks (m) minus the count of the most frequent type (max_freq).
        min_ops = m - max_freq

        return min_ops