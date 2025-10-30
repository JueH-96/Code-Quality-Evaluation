class Solution:
    def validStrings(self, n: int) -> List[str]:
        """
        Return all binary strings of length n such that
        no two consecutive characters are '0'.
        Equivalently, each length-2 substring has at least one '1'.
        """
        results = []
        
        def backtrack(current: str):
            # If we've reached length n, add to results
            if len(current) == n:
                results.append(current)
                return
            
            # We can always add '1'
            backtrack(current + '1')
            
            # We can add '0' only if the previous character isn't '0'
            if not current or current[-1] != '0':
                backtrack(current + '0')
        
        backtrack("")
        return results