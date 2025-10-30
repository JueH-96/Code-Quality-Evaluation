class Solution:
    def stringSequence(self, target: str) -> List[str]:
        res = []
        current = ""
        for c in target:
            current = current + 'a'
            res.append(current)
            k = ord(c) - ord('a')
            if k > 0:
                base = current[:-1]
                for j in range(1, k + 1):
                    new_char = chr(ord('a') + j)
                    new_str = base + new_char
                    res.append(new_str)
                current = base + c
        return res