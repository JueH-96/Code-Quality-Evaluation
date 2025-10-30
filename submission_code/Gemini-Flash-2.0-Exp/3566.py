class Solution:
    def stringSequence(self, target: str) -> List[str]:
        res = []
        curr = ""
        for char in target:
            while True:
                if not curr:
                    curr += 'a'
                    res.append(curr)
                elif curr[-1] == char:
                    break
                else:
                    curr = curr[:-1] + chr(((ord(curr[-1]) - ord('a') + 1) % 26) + ord('a'))
                    res.append(curr)
        return res