from collections import defaultdict
from typing import List

class Solution:
    def shortestSubstrings(self, arr: List[str]) -> List[str]:
        # Build a map from substring -> set of indices of strings in which it appears
        substr_map = defaultdict(set)
        for idx, s in enumerate(arr):
            seen = set()  # to avoid adding the same substring multiple times for this string
            n = len(s)
            for i in range(n):
                for l in range(1, n - i + 1):
                    sub = s[i:i+l]
                    if sub not in seen:
                        substr_map[sub].add(idx)
                        seen.add(sub)
        
        # For each string, find the shortest unique substring
        result = []
        for idx, s in enumerate(arr):
            n = len(s)
            answer = ""
            # try lengths from 1 up to full length
            for length in range(1, n+1):
                # collect all unique substrings of this length
                subs = {s[i:i+length] for i in range(n-length+1)}
                # test them in lex order
                for sub in sorted(subs):
                    # if this substring appears only in the current string
                    if substr_map[sub] == {idx}:
                        answer = sub
                        break
                if answer:
                    break
            result.append(answer)
        
        return result