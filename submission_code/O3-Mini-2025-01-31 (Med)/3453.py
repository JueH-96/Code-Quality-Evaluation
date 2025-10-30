class Solution:
    def validStrings(self, n: int) -> List[str]:
        result = []
        
        def backtrack(current: str):
            if len(current) == n:
                result.append(current)
                return
            # If the current string is empty or the last character is '1', we can add either '0' or '1'
            if not current or current[-1] == '1':
                backtrack(current + '0')
                backtrack(current + '1')
            # If the last character is '0', then we cannot add another '0' (to avoid "00")
            else:
                backtrack(current + '1')
        
        backtrack("")
        return result