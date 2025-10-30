from collections import defaultdict
from typing import List

class Solution:
    def shortestSubstrings(self, arr: List[str]) -> List[str]:
        substr_to_words = defaultdict(set)
        n = len(arr)
        
        # Preprocess all substrings and map them to their word indices
        for i, s in enumerate(arr):
            length = len(s)
            for j in range(length):
                for k in range(j + 1, length + 1):
                    sub = s[j:k]
                    substr_to_words[sub].add(i)
        
        res = []
        # Process each word to find the shortest unique substring
        for i, s in enumerate(arr):
            seen = set()
            candidates = []
            for j in range(len(s)):
                for k in range(j + 1, len(s) + 1):
                    sub = s[j:k]
                    if sub in seen:
                        continue
                    seen.add(sub)
                    if substr_to_words[sub] == {i}:
                        candidates.append(sub)
            
            if not candidates:
                res.append("")
                continue
            
            min_len = min(len(c) for c in candidates)
            min_candidates = [c for c in candidates if len(c) == min_len]
            min_candidates.sort()
            res.append(min_candidates[0])
        
        return res