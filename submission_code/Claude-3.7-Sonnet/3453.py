class Solution:
    def validStrings(self, n: int) -> List[str]:
        result = []
        
        def backtrack(current, index):
            if index == n:
                result.append(current)
                return
            
            # Add "1"
            backtrack(current + "1", index + 1)
            
            # Add "0" if it doesn't create "00"
            if index == 0 or current[-1] != "0":
                backtrack(current + "0", index + 1)
        
        # Start with an empty string
        backtrack("", 0)
        
        return result