from typing import List
from collections import defaultdict

class Solution:
    def shortestSubstrings(self, arr: List[str]) -> List[str]:
        def generate_substrings(s):
            substrings = set()
            n = len(s)
            for i in range(n):
                for j in range(i + 1, n + 1):
                    substrings.add(s[i:j])
            return substrings
        
        substring_count = defaultdict(int)
        
        for s in arr:
            substrings = generate_substrings(s)
            for sub in substrings:
                substring_count[sub] += 1
        
        result = []
        
        for s in arr:
            candidates = []
            substrings = generate_substrings(s)
            for sub in substrings:
                if substring_count[sub] == 1:
                    candidates.append(sub)
            
            if not candidates:
                result.append("")
            else:
                candidates.sort(key=lambda x: (len(x), x))
                result.append(candidates[0])
        
        return result