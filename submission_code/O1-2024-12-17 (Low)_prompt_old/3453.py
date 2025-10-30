class Solution:
    def validStrings(self, n: int) -> List[str]:
        from typing import List
        
        results = []
        
        def backtrack(path: str, last_char: str):
            if len(path) == n:
                results.append(path)
                return
            
            # We can always add '1'
            backtrack(path + '1', '1')
            
            # We can add '0' only if the last character was not '0'
            if last_char != '0':
                backtrack(path + '0', '0')
        
        # Start with an empty path
        backtrack("", "")
        return results