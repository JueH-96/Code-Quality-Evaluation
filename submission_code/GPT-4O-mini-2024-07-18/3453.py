from typing import List

class Solution:
    def validStrings(self, n: int) -> List[str]:
        result = []
        
        def backtrack(current):
            if len(current) == n:
                result.append(current)
                return
            
            # Add '0' if the last character is not '0' or if the length is less than 2
            if len(current) < 2 or current[-1] == '1':
                backtrack(current + '0')
            
            # Always can add '1'
            backtrack(current + '1')
        
        # Start backtracking with an empty string
        backtrack("")
        return result