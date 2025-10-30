class Solution:
    def validStrings(self, n: int) -> List[str]:
        def is_valid(s):
            for i in range(len(s) - 1):
                if s[i] == '0' and s[i+1] == '0':
                    return False
            return True

        result = []
        def backtrack(current_string):
            if len(current_string) == n:
                if is_valid(current_string):
                    result.append(current_string)
                return

            backtrack(current_string + "0")
            backtrack(current_string + "1")

        backtrack("")
        return result