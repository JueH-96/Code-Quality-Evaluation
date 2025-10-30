class Solution:
    def validStrings(self, n: int) -> List[str]:
        result = []
        
        def backtrack(s):
            if len(s) == n:
                result.append(s)
                return
            
            if not s or s[-1] == '1':
                # Can append either '0' or '1'
                backtrack(s + '0')
                backtrack(s + '1')
            else:  # s[-1] == '0'
                # Must append '1' to avoid "00"
                backtrack(s + '1')
        
        backtrack("")
        return result