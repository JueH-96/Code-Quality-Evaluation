from typing import List

class Solution:
    def validStrings(self, n: int) -> List[str]:
        if n == 1:
            return ["0", "1"]
        
        def backtrack(current):
            if len(current) == n:
                result.append(current)
                return
            if current[-1] == "0":
                backtrack(current + "1")
            else:
                backtrack(current + "0")
                backtrack(current + "1")
        
        result = []
        backtrack("0")
        backtrack("1")
        return result