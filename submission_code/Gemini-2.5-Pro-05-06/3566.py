from typing import List

class Solution:
  def stringSequence(self, target: str) -> List[str]:
    current_char_list = []  # Represents the string on screen as a list of characters
    result_strings = []     # Stores the sequence of strings appearing on screen

    # Iterate through each character of the target string.
    # For each character in `target`, we first extend the current string by 'a' (Key 1),
    # then modify this new 'a' to match the target character using Key 2 presses.
    for char_in_target in target:
      # --- Action for Key 1 ---
      # Press Key 1: appends 'a' to the string on screen.
      current_char_list.append('a')
      # Record the string formed after this key press.
      result_strings.append("".join(current_char_list))

      # --- Actions for Key 2 (if needed) ---
      # The last character is currently 'a'. We need to change it to `char_in_target`.
      # The number of Key 2 presses is the alphabetical distance from 'a'.
      # e.g., if char_in_target is 'a', 0 presses.
      # e.g., if char_in_target is 'c', current last char is 'a'.
      #   Press Key 2 (1st time): last char 'a' -> 'b'. String "...b". Add to results.
      #   Press Key 2 (2nd time): last char 'b' -> 'c'. String "...c". Add to results.
      # Total Key 2 presses = ord('c') - ord('a') = 2.
      
      num_key2_presses = ord(char_in_target) - ord('a')
      
      for _ in range(num_key2_presses):
        # Press Key 2: changes the last character to its next alphabetical character.
        # The problem states 'z' changes to 'a'. However, in this specific strategy,
        # we are always incrementing from 'a' towards `char_in_target`.
        # This sequence ('a' -> 'b' -> ... -> `char_in_target`) will not
        # require wrapping from 'z' to 'a'.
        # So, a simple ord(char) + 1 is sufficient.
        
        current_last_char_ord = ord(current_char_list[-1])
        current_char_list[-1] = chr(current_last_char_ord + 1)
        # Record the string formed after this key press.
        result_strings.append("".join(current_char_list))
            
    return result_strings