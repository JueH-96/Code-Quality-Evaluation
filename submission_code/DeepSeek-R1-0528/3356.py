from collections import defaultdict

class Solution:
    def shortestSubstrings(self, arr: List[str]) -> List[str]:
        occurrence = defaultdict(set)
        n = len(arr)
        for idx, s in enumerate(arr):
            l_len = len(s)
            for start in range(l_len):
                for end in range(start + 1, l_len + 1):
                    substr = s[start:end]
                    occurrence[substr].add(idx)
        
        res = [""] * n
        for i, s in enumerate(arr):
            l_len = len(s)
            for length in range(1, l_len + 1):
                candidates = []
                for start in range(l_len - length + 1):
                    substr = s[start:start + length]
                    if occurrence[substr] == {i}:
                        candidates.append(substr)
                if candidates:
                    res[i] = min(candidates)
                    break
        return res