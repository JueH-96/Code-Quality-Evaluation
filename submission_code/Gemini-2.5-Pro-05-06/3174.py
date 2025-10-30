class Solution:
  def minChanges(self, s: str) -> int:
    """
    Calculates the minimum number of changes to make a binary string beautiful.

    A string is beautiful if it can be partitioned into one or more substrings such that:
    - Each substring has an even length.
    - Each substring contains only 1's or only 0's.

    The reasoning above shows that it's always optimal to partition the string into
    substrings of length 2. Each such substring s[2k]s[2k+1] must be made
    monochromatic (either "00" or "11").
    - If s[2k] == s[2k+1], the pair is already monochromatic (e.g., "00" or "11"). 
      0 changes are needed for this pair.
    - If s[2k] != s[2k+1], the pair is "01" or "10". 
      1 change is needed to make it monochromatic (e.g., "01" can become "00" or "11").

    The algorithm iterates through the string s, considering pairs of characters
    s[2i]s[2i+1]. If s[2i] != s[2i+1], one change is counted for that pair.
    The total number of changes is the sum of changes for all such pairs.
    """
    
    changes = 0
    n = len(s)
    
    # Iterate through the string with a step of 2, processing pairs of characters.
    # s has an even length (guaranteed by constraints), so n is even.
    # The loop range i = 0, 2, ..., n-2 ensures s[i+1] is always a valid index.
    for i in range(0, n, 2):
      # s[i] and s[i+1] form a pair.
      if s[i] != s[i+1]:
        changes += 1
        
    return changes