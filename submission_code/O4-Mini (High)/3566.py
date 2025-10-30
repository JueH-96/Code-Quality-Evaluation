from typing import List

class Solution:
    def stringSequence(self, target: str) -> List[str]:
        res = []
        curr = ""
        for c in target:
            # Press key 1: append 'a'
            curr += "a"
            res.append(curr)
            # Press key 2: increment last char until it matches target character
            # Number of presses needed from 'a' to c
            delta = ord(c) - ord('a')
            for _ in range(delta):
                last = curr[-1]
                # wrap from 'z' to 'a'
                next_char = 'a' if last == 'z' else chr(ord(last) + 1)
                curr = curr[:-1] + next_char
                res.append(curr)
        return res