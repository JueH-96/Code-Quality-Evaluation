class Solution:
    def validStrings(self, n: int) -> List[str]:
        result = []
        
        def backtrack(current):
            if len(current) == n:
                result.append(current)
                return
            last_char = current[-1] if current else None
            if last_char == '0':
                backtrack(current + '1')
            elif last_char == '1':
                backtrack(current + '0')
                backtrack(current + '1')
            else:  # first character
                backtrack('0')
                backtrack('1')
        
        backtrack('')
        return result