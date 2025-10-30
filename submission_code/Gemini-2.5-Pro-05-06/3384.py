import collections

class Solution:
  def minimumOperationsToMakeKPeriodic(self, word: str, k: int) -> int:
    """
    Calculates the minimum number of operations to make a word k-periodic.

    Args:
      word: The input string.
      k: The length of the period. k divides word.length.

    Returns:
      The minimum number of operations.
    """
    n = len(word)
    
    # The word is divided into n/k blocks of length k.
    # For the word to be k-periodic, all these blocks must be identical.
    # An operation allows replacing one block (substring of length k starting at index i, i%k==0)
    # with a copy of another block (substring of length k starting at index j, j%k==0).
    #
    # To make all blocks identical, we should choose the most frequent block pattern
    # as the target pattern. All blocks that do not match this target pattern must be changed.
    # Let S_target be the most frequent block pattern. Let its frequency be max_freq.
    # This means max_freq blocks are already of the pattern S_target.
    # The remaining (num_blocks - max_freq) blocks need to be changed.
    # Each change (e.g., changing block B_p to pattern S_target) requires one operation.
    # This operation can be done by choosing an existing block B_q that has pattern S_target
    # and replacing B_p with B_q. Since S_target is an existing pattern (the most frequent one),
    # there is at least one such B_q (as long as num_blocks > 0).
    #
    # So, the minimum number of operations = num_blocks - max_freq.

    # Constraints check:
    # 1 <= n == word.length <= 10^5
    # 1 <= k <= word.length
    # k divides word.length.
    # word consists only of lowercase English letters.
    #
    # From these constraints, n >= 1 and k >= 1. Since k divides n, n/k is an integer >= 1.
    # So, num_blocks will always be at least 1.
    num_blocks = n // k

    # Use collections.Counter to efficiently store and update frequencies of block patterns.
    block_frequencies = collections.Counter()
    
    # Iterate through the word, extracting blocks of length k.
    # The loop iterates with a step of k, so `i` will take values 0, k, 2k, ..., n-k.
    # These are the starting indices of the blocks.
    for i in range(0, n, k):
        # Python string slicing word[start:end] extracts the substring
        # from index `start` up to (but not including) index `end`.
        # This creates a new string object for the block.
        block = word[i : i+k]
        block_frequencies[block] += 1
            
    # Since num_blocks >= 1 (guaranteed by constraints), the loop for `i`
    # will execute at least once (for i=0).
    # Therefore, `block_frequencies` will not be empty.
    # As a result, `block_frequencies.values()` will yield a non-empty sequence,
    # and calling `max()` on it will not raise an error.
    max_freq = 0
    # Check if block_frequencies is populated, which it will be due to constraints.
    if block_frequencies: 
        max_freq = max(block_frequencies.values())
    # If num_blocks is 0 (e.g. n=0, not allowed by constraints), then max_freq should be 0.
    # If num_blocks is >0, block_frequencies is non-empty, max_freq will be at least 1.

    # The number of operations required is the total number of blocks
    # minus the count of the most frequent block pattern.
    return num_blocks - max_freq