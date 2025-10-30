from typing import List

class Solution:
    def stringSequence(self, target: str) -> List[str]:
        result = []
        current = ""
        for c in target:
            base = current + 'a'
            result.append(base)
            diff = ord(c) - ord('a')
            for j in range(1, diff + 1):
                next_char = chr(ord('a') + j)
                new_str = current + next_char
                result.append(new_str)
            current += c
        return result