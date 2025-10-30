class Solution:
    def validStrings(self, n: int) -> List[str]:
        res = []
        def backtrack(current, last_char):
            if len(current) == n:
                res.append(current)
                return
            if last_char is None:
                # First step, can add '0' or '1'.
                backtrack(current + '0', '0')
                backtrack(current + '1', '1')
            else:
                if last_char == '0':
                    # Next must be '1'.
                    backtrack(current + '1', '1')
                else:
                    # Can be '0' or '1'.
                    backtrack(current + '0', '0')
                    backtrack(current + '1', '1')
        backtrack('', None)
        return res