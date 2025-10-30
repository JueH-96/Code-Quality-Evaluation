class Solution:
    def validStrings(self, n: int) -> List[str]:
        if n == 1:
            return ["0", "1"]
        
        result = []
        
        def backtrack(current_string):
            if len(current_string) == n:
                result.append(current_string)
                return
            
            # Try appending "0"
            if len(current_string) > 0 and current_string[-1] == "0":
                backtrack(current_string + "1")
            else:
                backtrack(current_string + "0")
                backtrack(current_string + "1")
        
        backtrack("0")
        backtrack("1")
        
        return result