from typing import List

class Solution:
    def stringSequence(self, target: str) -> List[str]:
        res = []
        prefix = ""  # current built string
        for ch in target:
            # 1) press key1: append 'a'
            prefix += 'a'
            res.append(prefix)
            # 2) press key2 as many times as needed to turn 'a' into ch
            steps = (ord(ch) - ord('a')) % 26
            # starting from 'a', do `steps` increments
            for i in range(steps):
                # build the next string by incrementing the last character
                new_char = chr(ord('a') + i + 1)
                prefix = prefix[:-1] + new_char
                res.append(prefix)
        return res