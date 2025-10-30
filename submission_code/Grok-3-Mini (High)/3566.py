from typing import List

class Solution:
    def stringSequence(self, target: str) -> List[str]:
        res = []
        curr_chars = []
        for i in range(len(target)):
            # Append 'a'
            curr_chars.append('a')
            res.append(''.join(curr_chars))
            # Number of increments needed
            D = ord(target[i]) - ord('a')
            for _ in range(D):
                # Increment the last character
                ord_last = ord(curr_chars[-1])
                new_ord = (ord_last - ord('a') + 1) % 26 + ord('a')
                curr_chars[-1] = chr(new_ord)
                res.append(''.join(curr_chars))
        return res