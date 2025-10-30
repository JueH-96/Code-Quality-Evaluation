class Solution:
    def validStrings(self, n: int) -> List[str]:
        def backtrack(remain, path):
            if remain == 0:
                res.append(path)
                return
            if path and path[-1] == '1':
                backtrack(remain - 1, path + '0')
            backtrack(remain - 1, path + '1')
            
        res = []
        backtrack(n, "")
        return res