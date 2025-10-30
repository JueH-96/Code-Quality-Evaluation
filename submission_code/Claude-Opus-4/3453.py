class Solution:
    def validStrings(self, n: int) -> List[str]:
        result = []
        
        def backtrack(current):
            # Base case: if we've built a string of length n
            if len(current) == n:
                result.append(current)
                return
            
            # Try adding '1' - always valid
            backtrack(current + '1')
            
            # Try adding '0' - only valid if previous char is not '0'
            if not current or current[-1] == '1':
                backtrack(current + '0')
        
        backtrack("")
        return result