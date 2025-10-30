from typing import List

class Solution:
    def stringSequence(self, target: str) -> List[str]:
        result = []
        current_s = ""

        def next_char(c):
            # Helper function to get the next character in the alphabet, wrapping 'z' to 'a'.
            # ord('a') is 97, ord('z') is 122.
            # Calculate the position of the current character (0-indexed)
            char_index = ord(c) - ord('a')
            # Calculate the position of the next character (0-indexed)
            next_char_index = (char_index + 1) % 26
            # Convert the next character position back to its ASCII value and then to character
            return chr(next_char_index + ord('a'))

        # The screen starts empty (""). The first possible key press is Key 1.
        # This adds 'a' to the screen. This is the first string in the sequence.
        current_s = "a"
        result.append(current_s)

        # Now, we need the first character to match target[0].
        # We use Key 2 repeatedly until the current single character string matches target[0].
        # We start from 'a' (which is already on screen).
        current_char = 'a'
        while current_char != target[0]:
            # Apply Key 2: Change the last character to the next one.
            # Since the string is just one character, this changes the entire string.
            current_char = next_char(current_char)
            current_s = current_char # Update the single-character string
            result.append(current_s) # Add the new string to the result list

        # Now, build the rest of the target string character by character (from index 1 onwards).
        for i in range(1, len(target)):
            target_char = target[i]

            # To add the next character, we must use Key 1 (append 'a').
            # This appends 'a' to the end of the current string (which is target[:i]).
            current_s = current_s + 'a'
            result.append(current_s) # Add the new string to the result list

            # The last character is now 'a'. We need to change it to target[i].
            # We use Key 2 repeatedly until the last character matches target[i].
            # We start changing from 'a' (the current last character).
            current_char = 'a'
            while current_char != target_char:
                # Apply Key 2: Change the last character.
                current_char = next_char(current_char)
                # Update the last character of the string. String slicing and concatenation.
                current_s = current_s[:-1] + current_char
                result.append(current_s) # Add the new string to the result list

        return result