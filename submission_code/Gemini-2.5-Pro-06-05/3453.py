from typing import List

class Solution:
    def validStrings(self, n: int) -> List[str]:
        """
        Generates all valid binary strings of length n.
        A string is valid if it does not contain the substring "00".
        """
        
        result = []
        
        def backtrack(path: list[str]):
            """
            Recursively builds valid strings character by character.
            """
            # Base case: If the current string has reached the desired length,
            # it's a valid solution.
            if len(path) == n:
                result.append("".join(path))
                return
            
            # To generate results in lexicographical order (as in the example),
            # we explore appending '0' before '1'.

            # Option 1: Try appending '0'.
            # This is valid only if the path is empty or the last character was '1'.
            if not path or path[-1] == '1':
                path.append('0')
                backtrack(path)
                path.pop()  # Backtrack to explore other possibilities
            
            # Option 2: Try appending '1'.
            # This is always a valid move.
            path.append('1')
            backtrack(path)
            path.pop()  # Backtrack

        # Start the backtracking process with an empty path.
        backtrack([])
        
        return result