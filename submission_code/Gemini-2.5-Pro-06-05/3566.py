from typing import List

class Solution:
  def stringSequence(self, target: str) -> List[str]:
    """
    Generates the sequence of strings appearing on screen to type the target
    using a special keyboard with minimum key presses.
    """
    result = []
    
    # current_chars represents the string on the screen as a list of characters.
    # This is more efficient for modification than using immutable strings.
    current_chars = []

    # We build the target string character by character. For each character
    # in the target, we perform a two-phase process that corresponds to the
    # minimal sequence of key presses.
    for target_char in target:
        # Phase 1: Extend the string by one character.
        # The only way to increase the string's length is to press Key 1,
        # which appends 'a'. The resulting string is recorded.
        current_chars.append('a')
        result.append("".join(current_chars))

        # Phase 2: Modify the newly appended 'a' to match the target character.
        # This is done by repeatedly pressing Key 2. We record all intermediate
        # strings created by these key presses.
        
        # We iterate through the character codes from the one after 'a' up to
        # the target character's code, simulating sequential presses of Key 2.
        # If target_char is 'a', this range is empty and the loop is skipped.
        for code in range(ord('a') + 1, ord(target_char) + 1):
            # Update the last character to the next one in the alphabet.
            current_chars[-1] = chr(code)
            # Record the new string that has appeared on the screen.
            result.append("".join(current_chars))
    
    return result