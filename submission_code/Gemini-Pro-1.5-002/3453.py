class Solution:
    def validStrings(self, n: int) -> List[str]:
        res = []

        def backtrack(current_string):
            if len(current_string) == n:
                res.append(current_string)
                return

            if len(current_string) > 0 and current_string[-1] == '0':
                backtrack(current_string + '1')
            else:
                backtrack(current_string + '0')
                backtrack(current_string + '1')
        
        backtrack("")
        
        ans = []
        for s in res:
            is_valid = True
            for i in range(n - 1):
                if s[i:i+2] == "00":
                    is_valid = False
                    break
            if is_valid:
                ans.append(s)
        return ans