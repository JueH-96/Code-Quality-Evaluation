class Solution:
    def validStrings(self, n: int) -> List[str]:
        if n == 1:
            return ["0", "1"]
        
        result = []
        
        def backtrack(s):
            if len(s) == n:
                result.append(s)
                return
            if s[-1] == "0":
                backtrack(s + "1")
            else:
                backtrack(s + "0")
                backtrack(s + "1")
        
        backtrack("0")
        backtrack("1")
        
        return result