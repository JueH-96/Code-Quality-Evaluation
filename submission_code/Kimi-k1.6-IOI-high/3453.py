class Solution:
    def validStrings(self, n: int) -> List[str]:
        result = []
        
        def backtrack(s, last_char):
            if len(s) == n:
                result.append(s)
                return
            if last_char == '0':
                backtrack(s + '1', '1')
            else:
                backtrack(s + '0', '0')
                backtrack(s + '1', '1')
        
        if n == 0:
            return []
        backtrack('0', '0')
        backtrack('1', '1')
        return result