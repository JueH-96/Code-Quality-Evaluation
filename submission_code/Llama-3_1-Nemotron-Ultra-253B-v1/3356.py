from collections import defaultdict
from typing import List

class Solution:
    def shortestSubstrings(self, arr: List[str]) -> List[str]:
        substr_map = defaultdict(set)
        for idx, s in enumerate(arr):
            len_s = len(s)
            for l in range(1, len_s + 1):
                for start in range(len_s - l + 1):
                    sub = s[start:start+l]
                    substr_map[sub].add(idx)
        
        answer = []
        for idx, s in enumerate(arr):
            ordered_subs = []
            len_s = len(s)
            for l in range(1, len_s + 1):
                current = []
                for start in range(len_s - l + 1):
                    current.append(s[start:start+l])
                current.sort()
                ordered_subs.extend(current)
            found = False
            for sub in ordered_subs:
                if len(substr_map[sub]) == 1:
                    answer.append(sub)
                    found = True
                    break
            if not found:
                answer.append("")
        return answer