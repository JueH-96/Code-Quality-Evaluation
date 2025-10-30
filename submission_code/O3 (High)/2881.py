from typing import List

class Solution:
    def splitWordsBySeparator(self, words: List[str], separator: str) -> List[str]:
        """
        For each string in `words`, split it using `separator`, discard empty
        tokens, and preserve the overall order of appearance.
        """
        res: List[str] = []

        for w in words:
            # str.split already handles multiple occurrences and returns
            # empty strings when the separator is at the edges or repeated.
            for part in w.split(separator):
                if part:                       # exclude empty strings
                    res.append(part)

        return res