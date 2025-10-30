from typing import List

class Solution:
    def validStrings(self, n: int) -> List[str]:
        results = [] # This list will store all valid binary strings.

        # The backtrack function constructs valid binary strings recursively.
        # index: The current position (0-indexed) in the string we are trying to build.
        # current_string: The binary string constructed so far.
        def backtrack(index: int, current_string: str):
            # Base case: If the current_string has reached the desired length 'n',
            # it is a valid string, so we add it to our results list.
            if index == n:
                results.append(current_string)
                return

            # Option 1: Append '1'
            # Appending '1' is always permissible because it cannot create the forbidden "00" substring.
            # (e.g., "0" + "1" = "01" is valid; "1" + "1" = "11" is valid).
            backtrack(index + 1, current_string + '1')

            # Option 2: Append '0'
            # Appending '0' is only allowed under specific conditions to avoid "00".
            # 1. If it's the very first character (index == 0), there's no preceding character
            #    to form "00" with. So, "0" is a valid start.
            # 2. If it's not the first character, the character just before the current position
            #    (i.e., the last character of `current_string`) must be '1'.
            #    This forms "10", which is a valid substring. If the last character was '0',
            #    appending another '0' would create "00", which is forbidden.
            if index == 0 or current_string[-1] == '1':
                backtrack(index + 1, current_string + '0')

        # Initiate the backtracking process.
        # We start at index 0 with an empty string.
        backtrack(0, "")
        
        return results