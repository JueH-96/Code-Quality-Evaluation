import math

class Solution:
  """
  Finds the minimum number of changes to make a binary string beautiful.

  A binary string is beautiful if it can be partitioned into substrings of even
  length, where each substring consists of only '0's or only '1's.

  Args:
    s: The input binary string of even length.

  Returns:
    The minimum number of changes required.
  """
  def minChanges(self, s: str) -> int:
    """
    Calculates the minimum changes by checking adjacent pairs.

    The problem states that a beautiful string can be partitioned into substrings
    of even length, with each substring containing only '0's or only '1's.
    Consider the smallest possible even length for these substrings, which is 2.
    Any beautiful partitioning can be refined into a partitioning where all
    substrings have length 2 (e.g., "0000" can be partitioned as "00|00").

    Therefore, we can consider partitioning the string `s` into non-overlapping
    substrings of length 2: s[0:2], s[2:4], ..., s[n-2:n].
    To make the entire string beautiful, we need to ensure each of these length-2
    substrings consists of identical characters (either "00" or "11").

    For each pair of characters `s[i]` and `s[i+1]`:
    - If `s[i] == s[i+1]` (e.g., "00" or "11"), the pair already meets the
      condition, and 0 changes are needed for this pair.
    - If `s[i] != s[i+1]` (e.g., "01" or "10"), we need exactly one change to
      make them identical (change "01" to "00" or "11", change "10" to "11" or "00").

    The minimum total number of changes for the entire string is the sum of the
    minimum changes required for each length-2 pair. We iterate through the
    string in steps of 2 and count how many pairs have differing characters.
    """
    n = len(s)
    changes = 0
    # Iterate through the string with a step of 2, examining pairs of characters.
    # The loop starts at index 0 and goes up to n-2 (inclusive), incrementing by 2 each time.
    for i in range(0, n, 2):
      # Check if the characters in the current pair (s[i] and s[i+1]) are different.
      if s[i] != s[i+1]:
        # If they are different, one change is required to make this pair uniform ("00" or "11").
        changes += 1
    # The total number of changes accumulated is the minimum required for the entire string.
    return changes