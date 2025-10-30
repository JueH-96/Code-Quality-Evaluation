from typing import List

class Solution:
    def shortestSubstrings(self, arr: List[str]) -> List[str]:
        n = len(arr)
        presence = {}
        for idx, s in enumerate(arr):
            m = len(s)
            for start in range(m):
                for end in range(start + 1, m + 1):
                    substr = s[start:end]
                    if substr not in presence:
                        presence[substr] = set()
                    presence[substr].add(idx)
        
        res = [""] * n
        for idx, s in enumerate(arr):
            m = len(s)
            found = False
            for l in range(1, m + 1):
                seen = set()
                candidates = []
                for start in range(0, m - l + 1):
                    substr = s[start:start + l]
                    if substr in seen:
                        continue
                    seen.add(substr)
                    if len(presence[substr]) == 1 and idx in presence[substr]:
                        candidates.append(substr)
                if candidates:
                    res[idx] = min(candidates)
                    found = True
                    break
        return res