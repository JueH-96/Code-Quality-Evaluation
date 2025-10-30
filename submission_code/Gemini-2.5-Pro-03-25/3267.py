import collections

class Solution:
  """
  Finds the length of the longest special substring occurring at least thrice.

  A special string consists of only one distinct character.
  A substring is a contiguous sequence of characters.
  """
  def maximumLength(self, s: str) -> int:
    """
    Calculates the maximum length of a special substring occurring at least thrice
    by analyzing contiguous blocks of identical characters.

    The logic considers three main ways a special substring `c*l` can occur at least 3 times:
    1. Within a single block of `c`'s of length `k1 >= l+2`.
       Example: "aaaaa" (k1=5). "aaa" (l=3) occurs 5-3+1=3 times. Max length is k1-2.
    2. Across the two longest blocks `k1 >= k2`.
       - If `k1 > k2`: Substring `c*k2` occurs `k1-k2+1 >= 2` times in block 1 and 1 time in block 2. Total >= 3. Max length is k2.
       - If `k1 == k2`: Substring `c*(k1-1)` occurs 2 times in block 1 and 2 times in block 2. Total = 4. Max length is k1-1. (Requires k1 >= 2).
    3. Across the three longest blocks `k1 >= k2 >= k3`.
       Substring `c*k3` occurs at least once in each block. Total >= 3. Max length is k3.

    Args:
      s: The input string consisting of lowercase English letters. Length is between 3 and 50.

    Returns:
      The length of the longest special substring occurring at least thrice,
      or -1 if no such substring exists.
    """
    # Use defaultdict to store lists of block lengths for each character.
    # e.g., for s = "aaabaaa", char_blocks = {'a': [3, 3], 'b': [1]}
    char_blocks = collections.defaultdict(list)
    n = len(s)
    
    # Step 1: Identify contiguous blocks of identical characters and record their lengths.
    # Constraints state 3 <= s.length <= 50, so no need to check for n=0.
    i = 0
    while i < n:
        current_char = s[i]
        j = i
        # Find the end index (exclusive) of the current block.
        while j < n and s[j] == current_char:
            j += 1
        # Calculate the length of the block.
        length = j - i
        # Add the length to the list for this character.
        char_blocks[current_char].append(length)
        # Continue scanning from the end of the current block.
        i = j

    # Step 2: Initialize the maximum length found. -1 indicates no qualifying substring found yet.
    max_len = -1

    # Step 3: Process each character that appeared in the string.
    for char in char_blocks:
        # Retrieve the list of block lengths for the current character.
        lengths = char_blocks[char]
        # Sort the lengths in descending order to easily access the longest blocks (k1 >= k2 >= k3 >= ...).
        lengths.sort(reverse=True)
        
        n_blocks = len(lengths)

        # Determine potential maximum lengths based on the distribution of block lengths.
        # These lengths correspond to special substrings (char * length) that occur at least 3 times.
        
        # Case 1: Contribution from the single longest block (k1).
        # A block of length k1 contains the substring char*(k1-2) exactly 3 times.
        # This requires k1 >= 3 for the length k1-2 to be at least 1.
        if n_blocks >= 1:
            k1 = lengths[0]
            if k1 >= 3:
                # Update max_len if k1-2 is greater.
                max_len = max(max_len, k1 - 2)

        # Case 2: Contribution from the top two longest blocks (k1, k2).
        if n_blocks >= 2:
            k1 = lengths[0]
            k2 = lengths[1]
            
            # If k1 > k2:
            # The substring char*k2 occurs k1-k2+1 times in block 1 and 1 time in block 2.
            # Total occurrences = k1-k2+2. Since k1 > k2, k1-k2 >= 1, so total >= 3.
            # The candidate length is k2. We need k2 >= 1 for a valid length.
            # max(max_len, k2) handles this correctly as lengths are positive.
            if k1 > k2:
                 max_len = max(max_len, k2)
                 
            # If k1 == k2:
            # The substring char*(k1-1) occurs 2 times in block 1 and 2 times in block 2.
            # Total occurrences = 4 >= 3.
            # The candidate length is k1-1. We need k1 >= 2 for the length k1-1 to be at least 1.
            else: # k1 == k2
                if k1 >= 2: 
                   # Update max_len if k1-1 is greater.
                   max_len = max(max_len, k1 - 1)

        # Case 3: Contribution from the top three longest blocks (k1, k2, k3).
        if n_blocks >= 3:
            # We already have k1, k2 from above if n_blocks >= 2. Get k3.
            k3 = lengths[2]
            # The substring char*k3 occurs at least once in each of the top three blocks
            # (>= k1-k3+1 in block 1, >= k2-k3+1 in block 2, >= 1 in block 3).
            # Total occurrences >= 1 + 1 + 1 = 3.
            # The candidate length is k3. We need k3 >= 1 for a valid length.
            # max(max_len, k3) handles this correctly.
            max_len = max(max_len, k3)
            
    # Step 4: Return the overall maximum length found across all characters.
    # If no candidate length was found that is >= 1, max_len remains -1.
    return max_len