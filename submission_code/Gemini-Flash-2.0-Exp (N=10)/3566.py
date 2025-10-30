class Solution:
    def stringSequence(self, target: str) -> List[str]:
        res = []
        curr = ""
        for char in target:
            while not curr or curr[-1] != char:
                if not curr:
                    curr += 'a'
                elif curr[-1] < char:
                    curr = curr[:-1] + chr(ord(curr[-1]) + 1)
                else:
                    curr = curr[:-1] + 'a'
                res.append(curr)
        return res