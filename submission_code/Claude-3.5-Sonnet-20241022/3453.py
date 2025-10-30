class Solution:
    def validStrings(self, n: int) -> List[str]:
        if n == 1:
            return ["0", "1"]
            
        def isValid(s):
            for i in range(len(s)-1):
                if s[i] == '0' and s[i+1] == '0':
                    return False
            return True
            
        def backtrack(curr, remaining):
            if remaining == 0:
                if isValid(curr):
                    result.append(curr)
                return
                
            backtrack(curr + "0", remaining - 1)
            backtrack(curr + "1", remaining - 1)
            
        result = []
        backtrack("", n)
        return result