from typing import List

class Solution:
    def stringSequence(self, target: str) -> List[str]:
        result = []
        current = ""
        # Process each character in the target one by one.
        for ch in target:
            if not current:  # For the very first letter, the screen is empty.
                # Key 1: Append "a" to the screen.
                current += "a"
                result.append(current)
                # Use Key 2 repeatedly until the last character becomes ch.
                while current[-1] != ch:
                    last = current[-1]
                    # If last character is 'z', wrap to 'a', otherwise increment by one.
                    new_char = 'a' if last == 'z' else chr(ord(last) + 1)
                    current = current[:-1] + new_char
                    result.append(current)
            else:
                # For subsequent letters, first press Key 1: append an "a".
                current = current + "a"
                result.append(current)
                # Then use Key 2 repeatedly until the last character becomes the desired letter.
                while current[-1] != ch:
                    last = current[-1]
                    new_char = 'a' if last == 'z' else chr(ord(last) + 1)
                    current = current[:-1] + new_char
                    result.append(current)
        return result