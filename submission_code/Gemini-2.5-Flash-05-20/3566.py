import collections
from typing import List

class Solution:
    def stringSequence(self, target: str) -> List[str]:
        # Initialize an empty string on the screen.
        current_screen = ""
        # Initialize the list to store all intermediate strings.
        result_strings = []

        # Iterate through each character of the target string.
        # 'k' represents the current index in 'target' that we are trying to match/form.
        for k in range(len(target)):
            # Step 1: Append 'a' to extend the string to the correct length.
            # This corresponds to pressing Key 1.
            # At this point, 'current_screen' holds the prefix target[0...k-1] (or the last
            # fully formed prefix). To form target[0...k], we must add a character.
            # The only way to add a character is to append 'a'.
            current_screen += 'a'
            result_strings.append(current_screen)

            # Step 2: Change the newly appended character (at index k) to target[k].
            # This corresponds to pressing Key 2 multiple times if necessary.
            
            # The character currently at index k (which was just appended as 'a' or modified in previous steps)
            char_at_k = current_screen[k] 
            # The target character we want to achieve at index k
            target_char = target[k]

            # Calculate the number of Key 2 presses needed.
            # The formula (ord(target_char) - ord(char_at_k) + 26) % 26 handles:
            #   - Positive differences (e.g., 'a' to 'c' is 2)
            #   - Negative differences (e.g., 'y' to 'a' is -24, then +26 gives 2, correctly wrapping around)
            num_presses = (ord(target_char) - ord(char_at_k) + 26) % 26

            # Apply Key 2 presses one by one, adding each intermediate string to the result.
            # Python strings are immutable, so to modify a character, we convert to a list of characters,
            # modify, and then join back into a string.
            for _ in range(num_presses):
                # Convert the current string on screen to a mutable list of characters.
                current_screen_list = list(current_screen)
                
                # Get the character at the specific index k that needs to be changed.
                char_to_modify = current_screen_list[k]
                
                # Calculate the ASCII value of the next character in the alphabet (cyclically).
                # (ord(char_to_modify) - ord('a')) gives a 0-25 value. Add 1 for next char,
                # then modulo 26 to handle 'z' wrapping to 'a'. Finally, add ord('a') to get ASCII.
                next_char_code = (ord(char_to_modify) - ord('a') + 1) % 26 + ord('a')
                
                # Update the character in the list.
                current_screen_list[k] = chr(next_char_code)
                
                # Convert the list back to a string and update current_screen.
                current_screen = "".join(current_screen_list)
                
                # Add the new string state to our results list.
                result_strings.append(current_screen)

        # Return the complete sequence of strings.
        return result_strings