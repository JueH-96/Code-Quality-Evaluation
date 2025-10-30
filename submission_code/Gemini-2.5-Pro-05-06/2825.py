class Solution:
  def minimizedStringLength(self, s: str) -> int:
    """
    Calculates the minimum length of a string after applying a specific operation any number of times.

    The operation:
    Choose an index i in the string, and let c be the character in position i.
    Delete the closest occurrence of c to the left of i (if any)
    and the closest occurrence of c to the right of i (if any).
    The character s[i] itself is NOT deleted by this operation.

    Based on the analysis:
    - If a character appears only once, it cannot be removed.
    - If a character appears multiple times, we can always reduce its count to one
      by repeatedly picking one instance of it and eliminating its neighbors of the same type.
    - Operations on one character type do not affect other character types.
    
    Therefore, the minimum length of the string is the number of distinct characters
    in the original string.
    """
    
    # Use a set to find all unique characters in the string s.
    # Python's set constructor takes an iterable (like a string) and
    # creates a set containing only the unique elements from that iterable.
    distinct_characters = set(s)
    
    # The length of the minimized string is the number of unique characters.
    return len(distinct_characters)