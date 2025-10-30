import collections

class Solution:
  """
  This class provides a solution to find the last non-empty string 
  before a string becomes empty through a specific character removal process,
  as described in the problem statement.
  """
  def lastNonEmptyString(self, s: str) -> str:
    """
    Finds the string s right before it becomes empty using the described operation.
    The operation involves removing the first occurrence of each character 'a' through 'z'
    repeatedly until the string is empty.

    The efficient approach determines the state of the string just before the last operation.
    This is achieved by first calculating the frequency of each character in the input string `s`.
    Let K be the maximum frequency found among all characters. The characters that will remain
    in the string just before the final operation are precisely the K-th occurrences of those
    characters whose total frequency in `s` is exactly K. These characters appear in the result 
    string in the same relative order as they appeared in the original string `s`.

    Args:
      s: The input string consisting of lowercase English letters. 
         Constraints: 1 <= s.length <= 5 * 10^5.

    Returns:
      The string s as it was right before the last operation that makes it empty.
    """

    # Step 1: Calculate frequency of each character in the input string s.
    # collections.Counter provides an efficient way to count character occurrences.
    counts = collections.Counter(s)
    
    # Step 2: Find the maximum frequency K among all characters present in s.
    # Since the constraints state s.length >= 1, the string is not empty, 
    # counts will contain at least one character, and max_freq will be at least 1.
    max_freq = 0
    # Iterate through the character counts derived from the input string 's'.
    # We only need to consider characters that actually appear in 's'.
    for char in counts:
      # Update max_freq if the current character's count is higher.
      if counts[char] > max_freq:
        max_freq = counts[char]
    
    # Step 3: Identify the set of characters that appear exactly K (max_freq) times.
    # These 'target' characters are the ones whose K-th instance will survive until the last step.
    target_chars = set()
    # Iterate through the character counts again to find all characters with frequency equal to max_freq.
    for char in counts:
      if counts[char] == max_freq:
        target_chars.add(char)
    
    # Step 4 & 5: Iterate through the original string s from left to right.
    # Keep track of the number of times each character has been encountered so far in this pass.
    # When we encounter the K-th instance of a character that belongs to target_chars,
    # we append it to our result builder. This preserves the relative order.
    
    result_builder = []
    # Use an array (list in Python) of size 26 to store the current counts for each lowercase letter 'a' through 'z'.
    # Initialize all counts to 0.
    current_counts_arr = [0] * 26 
    
    # Iterate through the input string s character by character.
    for char in s:
      # Calculate the 0-based index corresponding to the character (e.g., 'a' -> 0, 'b' -> 1, ..., 'z' -> 25).
      char_idx = ord(char) - ord('a')
      # Increment the count for the character encountered.
      current_counts_arr[char_idx] += 1
      
      # Check two conditions:
      # 1. Is the current character one of the target characters (those with frequency equal to max_freq)?
      # 2. Is this the K-th (max_freq) time we have encountered this character in this pass?
      if char in target_chars and current_counts_arr[char_idx] == max_freq:
        # If both conditions are true, this character instance is part of the final non-empty string.
        # Append it to the result_builder list.
        result_builder.append(char)
            
    # Step 6: Join the characters collected in the result_builder list to form the final result string.
    # The characters are already in the correct relative order due to the left-to-right pass through s.
    return "".join(result_builder)