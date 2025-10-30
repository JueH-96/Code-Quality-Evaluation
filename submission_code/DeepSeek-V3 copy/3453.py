class Solution:
    def validStrings(self, n: int) -> List[str]:
        if n == 1:
            return ["0", "1"]
        
        def backtrack(current):
            if len(current) == n:
                result.append(current)
                return
            last_char = current[-1] if current else ''
            if last_char == '1':
                backtrack(current + '0')
                backtrack(current + '1')
            else:
                backtrack(current + '1')
        
        result = []
        backtrack("")
        return result