import bisect
from collections import defaultdict

class Solution:
    def calculateScore(self, s: str) -> int:
        available = defaultdict(list)
        score = 0
        for i in range(len(s)):
            c = s[i]
            mirror_c = chr(ord('a') + ord('z') - ord(c))
            if mirror_c in available:
                lst = available[mirror_c]
                idx = bisect.bisect_left(lst, i)
                if idx > 0:
                    j = lst[idx - 1]
                    score += i - j
                    lst.pop(idx - 1)
                    continue  # Skip adding to available as i is used
            # Add current index to the available list for current character
            bisect.insort(available[c], i)
        return score