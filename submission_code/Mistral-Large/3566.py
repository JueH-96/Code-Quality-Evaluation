from typing import List

class Solution:
    def stringSequence(self, target: str) -> List[str]:
        result = []
        current = ""

        for char in target:
            if current == "":
                # If current string is empty, we need to start by pressing key 1
                current = "a"
                result.append(current)

            # Get the difference between the current last character and the target character
            diff = (ord(char) - ord(current[-1])) % 26

            # If the difference is 0, it means we need to add a new 'a' and then transform it
            if diff == 0:
                current += "a"
                result.append(current)
                diff = (ord(char) - ord('a')) % 26

            # Transform the last character to the target character
            for _ in range(diff):
                current = current[:-1] + chr((ord(current[-1]) - ord('a') + 1) % 26 + ord('a'))
                result.append(current)

        return result