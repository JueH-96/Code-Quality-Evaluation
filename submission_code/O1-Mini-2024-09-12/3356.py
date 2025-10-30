from typing import List
from collections import defaultdict

class Solution:
    def shortestSubstrings(self, arr: List[str]) -> List[str]:
        substring_map = defaultdict(set)
        for idx, s in enumerate(arr):
            n = len(s)
            for i in range(n):
                for j in range(i+1, n+1):
                    substring_map[s[i:j]].add(idx)
        
        answer = []
        for idx, s in enumerate(arr):
            n = len(s)
            found = False
            min_sub = ""
            for length in range(1, n+1):
                substrings = sorted([s[i:i+length] for i in range(n - length +1)])
                for substr in substrings:
                    if len(substring_map[substr]) == 1:
                        min_sub = substr
                        found = True
                        break
                if found:
                    break
            if found:
                answer.append(min_sub)
            else:
                answer.append("")
        return answer