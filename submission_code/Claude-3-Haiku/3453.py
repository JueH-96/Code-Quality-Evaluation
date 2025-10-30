class Solution:
    def validStrings(self, n: int) -> List[str]:
        def backtrack(curr, n):
            if len(curr) == n:
                result.append(curr)
                return
            
            for c in ['0', '1']:
                if len(curr) == 0 or (len(curr) >= 2 and curr[-2:] != '00'):
                    backtrack(curr + c, n)
        
        result = []
        backtrack('', n)
        return result