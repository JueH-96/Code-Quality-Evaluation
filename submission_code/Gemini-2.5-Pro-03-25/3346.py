import math # Not needed, abs is built-in. Retained here as placeholder if other math functions were used.

class Solution:
  """
  Solves the problem of finding the lexicographically smallest string t
  such that the cyclic distance between s and t is at most k.
  """
  def getSmallestString(self, s: str, k: int) -> str:
    """
    Finds the lexicographically smallest string t achievable from s
    by changing characters, such that the total distance cost is at most k.
    The distance between characters is measured cyclically ('a' to 'z').

    Args:
      s: The original string consisting of lowercase English letters.
      k: The maximum allowed total distance for changes.

    Returns:
      The lexicographically smallest string t that satisfies distance(s, t) <= k.
    """

    def min_dist(c1: str, c2: str) -> int:
      """
      Calculates the minimum cyclic distance between two lowercase English letters.
      For example, distance('a', 'c') is 2, and distance('a', 'z') is 1.
      """
      # Get the ASCII ordinal values of the characters.
      ord1 = ord(c1)
      ord2 = ord(c2)
      
      # Calculate the absolute difference in ordinal values.
      diff = abs(ord1 - ord2)
      
      # The cyclic distance is the minimum of the direct path along the alphabet
      # and the wrap-around path (total 26 letters in the cycle).
      # Example: 'a' to 'z'. diff = abs(ord('a') - ord('z')) = 25. 
      #          min(25, 26 - 25) = min(25, 1) = 1.
      # Example: 'a' to 'c'. diff = abs(ord('a') - ord('c')) = 2.
      #          min(2, 26 - 2) = min(2, 24) = 2.
      return min(diff, 26 - diff)

    # Use a list to efficiently build the characters of the resulting string t.
    result_chars_list = [] 
    # Keep track of the remaining distance budget.
    current_k = k 
    n = len(s)

    # Iterate through each character of the input string s from left to right.
    for i in range(n):
      # Get the original character at the current position.
      s_char = s[i] 

      # Optimization: If the remaining budget k is 0, no more changes can be made.
      # The rest of the resulting string must match the rest of the original string s.
      if current_k == 0:
        # Append the remaining part of the original string s to the result list.
        result_chars_list.append(s[i:]) 
        # Exit the loop as no further changes are possible.
        break

      # We want to find the lexicographically smallest character for the current position.
      # Iterate through all possible target characters starting from 'a'.
      found_best_char = False # Flag to ensure a character is always chosen
      for j in range(26): # Represents characters 'a' through 'z'
        target_char = chr(ord('a') + j)
        
        # Calculate the cost (distance) of changing s_char to target_char.
        cost = min_dist(s_char, target_char)

        # Check if this change is affordable with the current remaining budget k.
        if cost <= current_k:
          # Found the best (lexicographically smallest) character for this position.
          # Append it to the result list.
          result_chars_list.append(target_char)
          # Reduce the remaining budget by the cost incurred.
          current_k -= cost 
          found_best_char = True
          # Break the inner loop: we've found the optimal character for position i,
          # so move to the next position (i+1).
          break
      
      # This assertion serves as a logical guarantee check. Given k >= 0,
      # changing a character to itself has cost 0, which is always <= k.
      # Thus, the inner loop should always find a character and break.
      # assert found_best_char, "Internal logic error: Should always find a character."


    # Join the characters (and potentially the appended substring) in the list 
    # to form the final result string t.
    return "".join(result_chars_list)