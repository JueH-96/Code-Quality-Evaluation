import collections

class Solution:
  """
  Finds the length of the longest semi-repetitive substring in a given string of digits.
  A semi-repetitive string has at most one consecutive pair of identical digits.
  For example, "0010", "0123", "54944" are semi-repetitive, while "00101022", "111" are not.
  """
  def longestSemiRepetitiveSubstring(self, s: str) -> int:
    """
    Calculates the longest semi-repetitive substring length.

    Args:
      s: The input string of digits (0-9). Length is between 1 and 50.

    Returns:
      The length of the longest semi-repetitive substring within s.
    """
    n = len(s)
    # Constraints state 1 <= n <= 50.
    # Base case: If the string has length 1, the longest semi-repetitive substring
    # is the string itself, with length 1.
    if n <= 1:
      return n

    # Initialize the maximum length found so far.
    # Since n >= 1, any single character substring (length 1) is semi-repetitive.
    # So, the minimum possible answer is 1.
    max_len = 1

    # We use a nested loop approach (O(n^2)) which is efficient enough given n <= 50.
    # The outer loop iterates through all possible starting positions 'i' of a substring.
    for i in range(n):
      # For each starting position 'i', we extend the substring to the right (index 'j').
      # We keep track of the number of consecutive pairs within the current substring s[i...j].
      pair_count = 0

      # The inner loop iterates through possible ending positions 'j' (inclusive),
      # starting from i + 1 to form substrings of length 2 or more.
      # Substrings of length 1 (j=i) are implicitly handled by initializing max_len = 1.
      for j in range(i + 1, n):
        # Check if adding the character s[j] creates a consecutive pair with the previous character s[j-1].
        if s[j] == s[j-1]:
          pair_count += 1

        # Check if the current substring s[i...j] is semi-repetitive.
        # A substring is semi-repetitive if it has at most one pair (pair_count <= 1).
        if pair_count <= 1:
          # If it is semi-repetitive, calculate its length and update the maximum length found so far.
          current_len = j - i + 1
          max_len = max(max_len, current_len)
        else:
          # If pair_count becomes greater than 1 (i.e., 2 or more pairs), the substring s[i...j]
          # is no longer semi-repetitive. Any longer substring starting at 'i'
          # (e.g., s[i...k] where k > j) will also contain these pairs and thus won't be
          # semi-repetitive either. Therefore, we can stop extending the substring
          # starting at 'i' and move to the next starting position (i + 1).
          break # Break the inner loop (over j).

    # After checking all possible substrings, max_len holds the length of the longest semi-repetitive one.
    return max_len