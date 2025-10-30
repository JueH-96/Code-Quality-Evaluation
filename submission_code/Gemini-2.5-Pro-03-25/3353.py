import collections

class Solution:
  """
  Finds if any substring of length 2 in s is also present in the reverse of s.
  """
  def isSubstringPresent(self, s: str) -> bool:
    """
    Checks for the presence of a common length-2 substring between s and its reverse.

    Args:
      s: The input string.

    Returns:
      True if such a substring exists, False otherwise.
    """
    n = len(s)
    # Edge case: Cannot form a substring of length 2 if the string length is less than 2.
    if n < 2:
        return False

    # Reverse the string s.
    # Python's slicing [::-1] is an efficient way to do this.
    rev_s = s[::-1]

    # Iterate through all possible starting indices for a length-2 substring in the original string s.
    # The loop needs to go up to index n-2 to get the last substring s[n-2:n].
    for i in range(n - 1):
        # Extract the substring of length 2 starting at index i.
        # For example, if s = "abcde" and i = 0, substring = "ab".
        # If i = 1, substring = "bc", etc.
        substring = s[i : i + 2]

        # Check if this extracted substring is present anywhere within the reversed string.
        # The 'in' operator in Python efficiently checks for substring presence.
        if substring in rev_s:
            # If we find any such substring that exists in both s and its reverse,
            # we have met the condition and can immediately return True.
            return True

    # If the loop completes without finding any length-2 substring from s
    # that is also present in the reverse of s, it means no such substring exists.
    # In this case, we return False.
    return False