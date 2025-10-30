class Solution:
    def validStrings(self, n: int) -> List[str]:
        res = []
        def dfs(pos, prev_char, current_string):
            if pos == n:
                res.append(current_string)
                return
            if prev_char != '0':
                dfs(pos+1, '0', current_string+'0')
                dfs(pos+1, '1', current_string+'1')
            else:
                dfs(pos+1, '1', current_string+'1')
        dfs(0, '', '')
        return res