class Solution:
    def validStrings(self, n: int) -> List[str]:
        def backtrack(current, length):
            if length == n:
                result.append(current)
                return
            
            if length == 0 or current[-1] == '1':
                backtrack(current + '0', length + 1)
            backtrack(current + '1', length + 1)

        result = []
        if n == 1:
            return ['0', '1']
        
        backtrack('', 0)
        return result