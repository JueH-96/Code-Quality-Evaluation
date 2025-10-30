from typing import List

class Solution:
    def stringSequence(self, target: str) -> List[str]:
        res = []
        cur = ""
        for c in target:
            # Press key1: append 'a'
            cur += 'a'
            res.append(cur)
            # Press key2: increment last character until it equals c
            steps = ord(c) - ord('a')
            for _ in range(steps):
                # increment the last character
                last = cur[-1]
                # compute next character in alphabet, wrapping from 'z'->'a'
                nxt = chr((ord(last) - ord('a') + 1) % 26 + ord('a'))
                cur = cur[:-1] + nxt
                res.append(cur)
        return res