class Solution:
    def validStrings(self, n: int) -> List[str]:
        def is_valid(s: str) -> bool:
            for i in range(len(s) - 1):
                if s[i] == '0' and s[i + 1] == '0':
                    return False
            return True

        def backtrack(current: str):
            if len(current) == n:
                if is_valid(current):
                    result.append(current)
                return
            backtrack(current + '0')
            backtrack(current + '1')

        result = []
        backtrack("")
        return result