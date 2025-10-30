class Solution:
    def stringSequence(self, target: str) -> List[str]:
        res = []
        cur = ""
        for char in target:
            if not cur:
                cur += "a"
                res.append(cur)
            elif char == cur[-1]:
                cur += "a"
                res.append(cur)
            else:
                while cur[-1] != char:
                    cur = cur[:-1] + chr(ord(cur[-1]) + 1)
                    res.append(cur)
                    if ord(cur[-1]) == ord('z'):
                        cur = cur[:-1] + 'a'
                        res.append(cur)
                cur += "a"
                res.append(cur)
        return res