class Solution:
    def validStrings(self, n: int) -> List[str]:
        if n == 1:
            return ["0", "1"]
        res = ["0", "1"]
        for _ in range(2, n + 1):
            new_res = []
            for s in res:
                if s[-1] == '1':
                    new_res.append(s + '0')
                    new_res.append(s + '1')
                else:
                    new_res.append(s + '1')
            res = new_res
        return res