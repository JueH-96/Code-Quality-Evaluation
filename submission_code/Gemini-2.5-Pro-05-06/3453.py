from typing import List

class Solution:
  def validStrings(self, n: int) -> List[str]:
    results: List[str] = []
    # current_path_chars stores the characters of the binary string being built.
    # Using a list of characters (List[str]) is efficient for append/pop operations.
    current_path_chars: List[str] = []

    # backtrack_helper is a nested function that uses closures to access 
    # results, current_path_chars, and n from the outer scope.
    def backtrack_helper():
      # Base case: If the current path has reached the desired length n
      if len(current_path_chars) == n:
        # Join the characters to form a string and add to results.
        # e.g., if current_path_chars = ['0', '1', '0'], "".join(...) -> "010"
        results.append("".join(current_path_chars))
        return

      # Recursive step: Try appending '0' or '1' to extend the current_path_chars

      # Option 1: Try appending '0'
      # A '0' can be appended if:
      #   - The current path is empty (current_path_chars is empty), meaning '0' would be the first character.
      #   - Or, the last character already in current_path_chars was '1'.
      # This condition prevents forming the substring "00".
      if not current_path_chars or current_path_chars[-1] == '1':
        current_path_chars.append('0')
        backtrack_helper()
        # After the recursive call returns (all paths starting with current_path_chars + '0' explored),
        # remove '0' to backtrack and explore other options (like appending '1' next).
        current_path_chars.pop() 

      # Option 2: Try appending '1'
      # A '1' can always be appended. If the previous character was '0', it forms "01".
      # If the previous character was '1', it forms "11". Neither violates the "no 00" rule.
      current_path_chars.append('1')
      backtrack_helper()
      # Backtrack: remove '1' after its branch is fully explored.
      current_path_chars.pop()

    # Initial call to the helper function to start the generation process.
    # current_path_chars is initially empty.
    backtrack_helper()
    
    return results