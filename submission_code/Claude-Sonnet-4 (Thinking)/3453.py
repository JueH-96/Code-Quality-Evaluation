class Solution:
    def validStrings(self, n: int) -> List[str]:
        result = []
        
        def backtrack(current_string):
            if len(current_string) == n:
                result.append(current_string)
                return
            
            # Try to place "0" first (only if valid)
            if not current_string or current_string[-1] != "0":
                backtrack(current_string + "0")
            
            # Can always place "1"
            backtrack(current_string + "1")
        
        backtrack("")
        return result