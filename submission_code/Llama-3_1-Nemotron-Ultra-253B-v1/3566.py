from typing import List

class Solution:
    def stringSequence(self, target: str) -> List[str]:
        result = []
        current = []
        for c in target:
            # Append 'a' and add to result
            current.append('a')
            result.append(''.join(current))
            # Calculate the number of increments needed
            increments = ord(c) - ord('a')
            for _ in range(increments):
                # Increment the last character
                last_char = current[-1]
                new_char = chr((ord(last_char) - ord('a') + 1) % 26 + ord('a'))
                current[-1] = new_char
                result.append(''.join(current))
        return result